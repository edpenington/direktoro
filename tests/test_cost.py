"""Tests for the model registry accessors (direktoro.registry) and the cost
arithmetic (direktoro.cost).

Two properties, and they are complementary. The registry resolves an id to how
a model is reached and what it can do, and an unknown id fails loudly rather
than being guessed at. The cost path prices usage against rates the CALLER
supplies and refuses to price anything it was given no rate for, so a total is
never quietly too low.

Rates themselves live in `direktoro.prices` (tested in `tests/test_prices.py`),
where each one is dated and carries its source; `TestEveryPriceLivesInOnePlace`
is the standing guard that they stay there.
"""

import pytest

from direktoro.cost import COUNTERS, cost_from_rates, cost_from_usage
from direktoro.providers import NormalisedUsage, resolved_decoding_params
from direktoro.registry import (
    MODEL_REGISTRY, Model, SAMPLING_PARAMS, is_known_model, is_retired,
    known_models, model_info, sampling_band)


class TestKnownModels:
    def test_registry_is_the_registry_keys(self):
        assert known_models() == sorted(MODEL_REGISTRY)

    def test_is_known_model_true_for_registered(self):
        assert is_known_model("claude-sonnet-4-6")

    def test_is_known_model_false_for_unknown(self):
        assert not is_known_model("totally-made-up-model-9000")

    def test_direct_provider_models_are_known(self):
        # Anthropic and OpenAI stay direct.
        for m in ("claude-opus-4-8", "gpt-5.6-sol", "gpt-5.6-terra"):
            assert is_known_model(m)

    def test_routed_models_are_known(self):
        # GLM / Qwen route through OpenRouter; the registry id is the
        # OpenRouter slug verbatim.
        for m in ("z-ai/glm-5v-turbo", "z-ai/glm-4.6v",
                  "qwen/qwen3-vl-235b-a22b-instruct"):
            assert is_known_model(m)

    def test_a_routed_model_is_registered_only_under_its_gateway_slug(self):
        # A routed entry's id IS the OpenRouter slug, vendor prefix and all
        # (id-as-identity), and the bare model name is deliberately NOT also a
        # key. Two ids for one model would let a config name the one whose
        # serving arrangement it did not mean, and they are not
        # interchangeable: only the slug carries the pin, the ZDR discipline
        # and the served-upstream assertion. So the bare form fails registry
        # validation rather than quietly resolving to a differently-served
        # model. Derived from the table, so a new routed entry is covered
        # without anyone adding it here.
        for model_id, info in MODEL_REGISTRY.items():
            if info.route is None:
                continue
            assert "/" in model_id, model_id
            assert not is_known_model(model_id.split("/", 1)[1]), model_id

    def test_an_id_absent_from_the_table_is_simply_unknown(self):
        # The other half of the deletion rule (see the registry's DELETING AN
        # ENTRY note): an id is deleted rather than flagged retired only when
        # no stored run depends on it resolving, and once deleted it is not
        # special — naming it fails registry validation at startup, which is
        # earlier and louder than the provider's 404 mid-run.
        assert not is_known_model("claude-3-5-haiku-20241022")

    def test_legacy_snapshots_with_runs_behind_them_stay_resolvable(self):
        # The first half of the same rule: these three are retired upstream
        # (see TestRetiredFlag) but flagged rather than deleted, so a run that
        # named one still resolves and can still be cited. Guards against an
        # over-broad deletion.
        for m in ("claude-3-5-sonnet-20241022", "claude-sonnet-4-20250514",
                  "claude-opus-4-20250514"):
            assert is_known_model(m)


class TestRetiredFlag:
    """`Model.retired` marks a provider-withdrawn id. A retired id must still
    RESOLVE for provenance; it is rejected wherever a caller applies the
    new-run gate (`is_retired` / `known_models(include_retired=False)`,
    exercised in TestRetirementGateIsReachableFromTheLibrary below, and applied
    by direktoro's own CLI in test_cli.py)."""

    # The Anthropic ids withdrawn upstream, with their retirement dates
    # (deprecation table verified 2026-07-31). Flagged rather than deleted so
    # past runs still cite; a caller applying the gate refuses them for new
    # runs.
    RETIRED_ANTHROPIC_IDS = {
        "claude-3-5-sonnet-20241022": "2025-10-28",
        "claude-sonnet-4-20250514": "2026-06-15",
        "claude-opus-4-20250514": "2026-06-15",
    }

    def test_field_defaults_false(self):
        # A minimal Model construction carries retired False, so the flag is
        # opt-in per entry. (forced_tool_choice has no default and must be
        # stated even here.)
        assert Model("anthropic", None, "ANTHROPIC_API_KEY",
                     supports_images=True,
                     forced_tool_choice=True).retired is False

    def test_withdrawn_ids_are_flagged_retired(self):
        for model_id in self.RETIRED_ANTHROPIC_IDS:
            assert model_info(model_id).retired is True, model_id

    def test_no_other_entry_is_retired(self):
        # Guard against the flag spreading to a live model, which would refuse
        # new runs on a model that still works.
        for model_id, m in MODEL_REGISTRY.items():
            if model_id in self.RETIRED_ANTHROPIC_IDS:
                continue
            assert m.retired is False, model_id

    def test_retired_entries_still_resolve(self):
        # The whole point of flagging rather than deleting: a run that already
        # happened on one of these still resolves to its provider and wire.
        for model_id in self.RETIRED_ANTHROPIC_IDS:
            assert is_known_model(model_id)
            assert model_info(model_id).provider == "anthropic"

    def test_a_retired_entry_resolves_like_a_live_one(self, monkeypatch):
        # The registry's dual role must survive the flag: a retired id is still
        # looked up for provenance, and it is only NEW runs that reject it. So
        # model_info must NOT raise on one.
        retired = Model("anthropic", None, "ANTHROPIC_API_KEY",
                        supports_images=True,
                        forced_tool_choice=True, retired=True)
        monkeypatch.setitem(MODEL_REGISTRY, "synthetic-retired-1", retired)

        assert is_known_model("synthetic-retired-1")
        assert model_info("synthetic-retired-1").retired is True
        assert model_info("synthetic-retired-1").api_key_env == \
            "ANTHROPIC_API_KEY"


class TestRetirementGateIsReachableFromTheLibrary:
    """A consumer that never runs the CLI must still be able to ask whether an
    id can start a new run.

    Every accessor here resolves a retired id — deliberately, because the table
    is also the provenance record — so a library caller applying no gate reaches
    the provider and gets a 404 on a paid path. Making the gate the CLI's
    private business would leave that consumer with no way to ask the question
    at all, which is why the predicate and the narrowed list are part of the
    public surface rather than an internal detail of `resolve_models`."""

    RETIRED_ID = "claude-sonnet-4-20250514"
    LIVE_ID = "claude-opus-5"

    def test_is_retired_answers_for_both(self):
        assert is_retired(self.RETIRED_ID) is True
        assert is_retired(self.LIVE_ID) is False

    def test_is_retired_agrees_with_the_record_for_every_entry(self):
        # The predicate is the record, not a second opinion that can drift.
        for model_id, info in MODEL_REGISTRY.items():
            assert is_retired(model_id) is info.retired, model_id

    def test_is_retired_raises_on_an_unknown_id(self):
        # An id that is not in the table at all must not answer a reassuring
        # False: "not retired" would read as "safe to run".
        with pytest.raises(ValueError, match="unknown model"):
            is_retired("totally-made-up-model-9000")

    def test_known_models_still_lists_retired_ids_by_default(self):
        # The default stays the PROVENANCE list. Narrowing it would silently
        # drop ids from every consumer's lookup on upgrade — the exact failure
        # the retired flag exists to avoid.
        assert known_models() == sorted(MODEL_REGISTRY)
        assert self.RETIRED_ID in known_models()

    def test_known_models_can_return_only_the_startable_ids(self):
        startable = known_models(include_retired=False)
        assert self.RETIRED_ID not in startable
        assert self.LIVE_ID in startable
        assert startable == sorted(
            model_id for model_id, info in MODEL_REGISTRY.items()
            if not info.retired)

    def test_the_two_forms_of_the_gate_agree(self):
        startable = set(known_models(include_retired=False))
        assert startable == {model_id for model_id in known_models()
                             if not is_retired(model_id)}

    def test_every_accessor_still_resolves_a_retired_id(self):
        # The gate must stay OUTSIDE the lookups: a run that already happened
        # on a withdrawn id has to keep resolving, or its stored record stops
        # being interpretable. This is what makes an exported gate necessary
        # rather than optional.
        from direktoro import (
            call_identity_fields, model_supports_images,
            supports_forced_tool_choice, rejected_sampling_params,
            thinking_support)

        assert model_info(self.RETIRED_ID).provider == "anthropic"
        assert model_supports_images(self.RETIRED_ID) is True
        assert supports_forced_tool_choice(self.RETIRED_ID) is True
        assert rejected_sampling_params(self.RETIRED_ID) == frozenset()
        assert thinking_support(self.RETIRED_ID) is None
        assert call_identity_fields(self.RETIRED_ID)["model"] == self.RETIRED_ID


class TestUnknownIdErrorSeparatesLiveFromRetired:
    """The unknown-id error is also a suggestion list, so it must not suggest a
    withdrawn id as a fix for a typo — that trades an error from here for a 404
    from the provider. Retired ids are still named, because they are the right
    answer when the question is about a past run, but they are named as not
    startable."""

    RETIRED_ID = "claude-sonnet-4-20250514"

    def _message(self):
        with pytest.raises(ValueError) as excinfo:
            model_info("totally-made-up-model-9000")
        return str(excinfo.value)

    def test_the_startable_ids_are_listed_first_and_unqualified(self):
        message = self._message()
        assert "claude-opus-5" in message
        head = message.split("RETIRED")[0]
        assert "claude-opus-5" in head

    def test_a_retired_id_is_listed_but_marked_not_startable(self):
        message = self._message()
        assert self.RETIRED_ID in message
        assert "RETIRED" in message
        # And it is not offered among the ids available for a new run.
        assert self.RETIRED_ID not in message.split("RETIRED")[0]


class TestModelFieldOrder:
    """`Model` is constructed POSITIONALLY, so a new field must be APPENDED.

    Every entry in `MODEL_REGISTRY` passes provider / base_url / api_key_env
    positionally, so do the tests here, and so does any caller that builds a
    synthetic entry. Inserting a field in the middle rebinds all of those to the
    attribute after it — silently, with no TypeError, because the arity is
    unchanged: a base URL lands in `api_key_env`, a key env lands in `quirks`,
    and the registry keeps importing. Appending is the only edit that cannot do
    that.

    So the order is pinned. A field added at the end extends the list here by
    one; a field added anywhere else fails, which is the point. The `Model`
    docstring states the rule, and this is what makes forgetting it loud."""

    FIELD_ORDER = (
        # Positional in every construction. Reordering these is the breakage.
        "provider",
        "base_url",
        "api_key_env",
        # Defaulted, and passed by keyword — but still positionally reachable,
        # so they are pinned too.
        "quirks",
        "wire_api",
        "supports_images",
        "forced_tool_choice",
        "rejects_sampling",
        "retired",
        "route",
        "thinking",
        "sampling_bands",
    )

    def test_field_order_is_pinned(self):
        import dataclasses

        actual = tuple(f.name for f in dataclasses.fields(Model))
        assert actual == self.FIELD_ORDER, (
            "Model's field order changed. A new field must be APPENDED at the "
            "end with a default: this record is constructed positionally, so "
            "inserting one mid-list rebinds every later positional argument to "
            "the wrong attribute without raising. If you appended, add the new "
            "name to the END of FIELD_ORDER.")

    def test_the_three_positional_fields_bind_where_they_are_read(self):
        # The failure the order guards against, made concrete: these three are
        # what every registry entry passes positionally, and what the adapters
        # read back to reach a model.
        entry = Model("anthropic", "https://example.invalid/v1", "SOME_KEY_ENV",
                      supports_images=True,
                      forced_tool_choice=True)
        assert entry.provider == "anthropic"
        assert entry.base_url == "https://example.invalid/v1"
        assert entry.api_key_env == "SOME_KEY_ENV"

    def test_every_field_after_the_third_has_a_default(self):
        # What keeps appending safe: an appended field with no default would be
        # a required fourth positional argument, breaking every construction.
        import dataclasses

        for f in dataclasses.fields(Model)[3:]:
            has_default = (f.default is not dataclasses.MISSING
                           or f.default_factory is not dataclasses.MISSING)
            assert has_default, (
                f"Model.{f.name} has no default, so it is a required "
                f"positional argument and every existing Model(...) call is "
                f"now a TypeError. Give it a default.")


class TestForcedToolChoiceIsStated:
    """The flag has no working default: whether an endpoint honours a forced
    tool_choice was either established or it was not, so every entry states it
    and an unstated one is a construction error, not a silent claim."""

    def test_unstated_flag_is_refused_at_construction(self):
        # supports_images is stated so the construction fails on THIS flag and
        # not on the other sentinel — the two are checked independently.
        with pytest.raises(ValueError, match="forced_tool_choice"):
            Model("anthropic", None, "ANTHROPIC_API_KEY",
                  supports_images=True)

    @pytest.mark.parametrize("value", ["yes", "false", 0, 1, [True]])
    def test_a_non_bool_flag_is_refused(self, value):
        # The record is reachable positionally, and a truthy non-bool read as
        # True would force a named tool on an endpoint that 404s one — the
        # paid failure the statement requirement exists to prevent.
        with pytest.raises(ValueError, match="as a bool"):
            Model("anthropic", None, "ANTHROPIC_API_KEY",
                  supports_images=True,
                  forced_tool_choice=value)

    def test_every_registry_entry_states_the_flag(self):
        # Import already enforces this (a None would have raised); the assert
        # documents that the table holds real booleans, not sentinels.
        for model_id, m in MODEL_REGISTRY.items():
            assert isinstance(m.forced_tool_choice, bool), model_id

    def test_exactly_two_fields_require_statement(self):
        # The append-only rule's other half used to be "every field after the
        # third has a default that WORKS". Two fields deliberately trade that
        # away — supports_images and forced_tool_choice, the two whose default
        # would read as a capability CLAIM — and this pins the trade to
        # exactly those: a construction stating only the three positionals
        # plus both flags succeeds, so no third sentinel has crept in.
        m = Model("anthropic", None, "ANTHROPIC_API_KEY",
                  supports_images=True,
                  forced_tool_choice=True)
        assert m.retired is False
        assert m.rejects_sampling == frozenset()
        assert m.thinking is None


class TestSupportsImagesIsStated:
    """The flag has no working default either, and for the costlier reason: a
    default of True would make a text-only entry that nobody thought about
    report as vision-capable, so a consumer sends image parts and is billed for
    the rejection. Which inputs an endpoint takes is part of registering it."""

    def test_unstated_flag_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="supports_images"):
            Model("anthropic", None, "ANTHROPIC_API_KEY",
                  forced_tool_choice=True)

    @pytest.mark.parametrize("value", ["yes", "false", 0, 1, [True]])
    def test_a_non_bool_flag_is_refused(self, value):
        # Same positional exposure as forced_tool_choice: a truthy non-bool
        # read as True is the paid 400 the statement requirement prevents.
        with pytest.raises(ValueError, match="as a bool"):
            Model("anthropic", None, "ANTHROPIC_API_KEY",
                  supports_images=value, forced_tool_choice=True)

    def test_every_registry_entry_states_the_flag(self):
        # Import already enforces this (a None would have raised); the assert
        # documents that the table holds real booleans, not sentinels — which
        # is what lets a consumer gate a pipeline on the value.
        for model_id, m in MODEL_REGISTRY.items():
            assert isinstance(m.supports_images, bool), model_id

    def test_a_text_only_entry_is_expressible_and_reports_as_such(self):
        # The whole point of removing the default: False must be a value the
        # table can hold and an accessor can report, so a consumer that
        # refuses a text-only model has something to refuse on.
        from direktoro import model_supports_images

        text_only = Model("anthropic", None, "ANTHROPIC_API_KEY",
                          supports_images=False, forced_tool_choice=True)
        assert text_only.supports_images is False
        import pytest as _pytest
        with _pytest.MonkeyPatch.context() as mp:
            mp.setitem(MODEL_REGISTRY, "synthetic-text-only", text_only)
            assert model_supports_images("synthetic-text-only") is False


class TestSamplingBands:
    """A band records the documented (low, high) range for one sampling param.
    Absence is honest — nothing established — and a band can never contradict
    `rejects_sampling`, which claims the param is refused outright."""

    def test_documented_band_is_returned(self):
        # Anthropic documents temperature 0.0-1.0; the entries that still take
        # sampling carry it.
        assert sampling_band("claude-sonnet-4-6", "temperature") == (0.0, 1.0)
        assert sampling_band("claude-haiku-4-5-20251001",
                             "temperature") == (0.0, 1.0)

    def test_unestablished_band_is_none(self):
        # No numeric range is published for Anthropic's top_p / top_k, so no
        # band is declared — the value is sent and the endpoint answers.
        assert sampling_band("claude-sonnet-4-6", "top_p") is None
        assert sampling_band("claude-sonnet-4-6", "top_k") is None

    def test_rejected_param_has_no_band(self):
        # Opus 5 refuses temperature outright; a band for it would be a
        # contradiction, so the accessor reports nothing.
        assert sampling_band("claude-opus-5", "temperature") is None

    def test_every_routed_entry_is_accounted_for(self):
        # The band on a routed entry is OpenRouter's own documented request
        # range, not an upstream fact a probe could establish — and the two
        # Gemini entries, which refuse every sampling control outright,
        # deliberately carry none. Derived from the table so a new routed entry
        # must take a position here, and the exceptions are NAMED rather than
        # read off `rejects_sampling` so that taking the position stays a
        # deliberate act instead of something an entry falls into.
        for model_id, info in MODEL_REGISTRY.items():
            if info.route is None:
                continue
            if model_id in ("google/gemini-3.6-flash",
                            "google/gemini-3.7-flash"):
                assert not info.sampling_bands, model_id
                continue
            assert sampling_band(model_id, "temperature") == (0.0, 2.0)
            assert sampling_band(model_id, "top_p") == (0.0, 1.0)

    def test_undeclared_bands_stay_undeclared(self):
        # The GPT entries' comment says no band is declared because their
        # accepted ranges were not re-read; pinned so a later hand adding one
        # has to bring the read with it. The retired entries likewise carry
        # none — no current reference describes them.
        for model_id in ("gpt-5.6-sol", "gpt-5.6-terra",
                         "claude-sonnet-4-20250514",
                         "claude-3-5-sonnet-20241022",
                         "claude-opus-4-20250514"):
            assert not model_info(model_id).sampling_bands, model_id

    def test_bands_are_read_only(self):
        # A band is a recorded documented fact; handing out a mutable mapping
        # would let one reader edit every later reader's copy of the record.
        bands = model_info("claude-sonnet-4-6").sampling_bands
        with pytest.raises(TypeError):
            bands["temperature"] = (0.0, 99.0)

    @pytest.mark.parametrize("bad", [None, {"temperature"},
                                     [("temperature", (0.0, 1.0))]])
    def test_a_non_mapping_container_is_refused(self, bad):
        with pytest.raises(TypeError, match="must be a dict"):
            Model("anthropic", None, "K", supports_images=True,
            forced_tool_choice=True,
                  sampling_bands=bad)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="unknown model"):
            sampling_band("totally-made-up-model-9000", "temperature")

    def test_unknown_param_name_refused_at_construction(self):
        with pytest.raises(ValueError, match="not a sampling"):
            Model("anthropic", None, "K", supports_images=True,
            forced_tool_choice=True,
                  sampling_bands={"presence_penalty": (0.0, 1.0)})

    def test_band_contradicting_a_refusal_is_refused(self):
        with pytest.raises(ValueError, match="rejects_sampling"):
            Model("anthropic", None, "K", supports_images=True,
            forced_tool_choice=True,
                  rejects_sampling=frozenset({"temperature"}),
                  sampling_bands={"temperature": (0.0, 1.0)})

    @pytest.mark.parametrize("band", [
        (1.0,),                       # not a pair
        [0.0, 1.0],                   # not a tuple
        (1.0, 0.0),                   # reversed
        ("0.0", "1.0"),               # not numbers
        (False, True),                # booleans are not band values
        (float("nan"), 1.0),          # NaN satisfies low <= high vacuously
        (float("nan"), float("nan")),
        (float("-inf"), float("inf")),  # declares a fact, claims nothing
    ])
    def test_malformed_band_is_refused(self, band):
        with pytest.raises(ValueError, match="low, high"):
            Model("anthropic", None, "K", supports_images=True,
            forced_tool_choice=True,
                  sampling_bands={"temperature": band})

    def test_out_of_band_value_is_refused_before_spend(self):
        # Anthropic documents temperature 0.0-1.0, so 1.5 — which a
        # cross-provider union bound would pass to the endpoint to 400 on a
        # paid call — is refused by the resolver.
        with pytest.raises(ValueError, match="outside the range"):
            resolved_decoding_params(
                "claude-sonnet-4-6", sampling={"temperature": 1.5},
                max_tokens=4096)

    def test_in_band_values_pass_and_the_band_is_per_model(self):
        dec = resolved_decoding_params(
            "claude-sonnet-4-6", sampling={"temperature": 1.0},
            max_tokens=4096)
        assert dec["temperature"] == 1.0
        # The same 1.5 is fine on the gateway surface, whose documented range
        # runs to 2.0 — which is the whole point of a per-model band.
        dec = resolved_decoding_params(
            "z-ai/glm-4.6v", sampling={"temperature": 1.5}, max_tokens=4096)
        assert dec["temperature"] == 1.5
        with pytest.raises(ValueError, match="outside the range"):
            resolved_decoding_params(
                "z-ai/glm-4.6v", sampling={"temperature": 2.5},
                max_tokens=4096)

    def test_a_param_with_no_band_passes_through(self):
        # No range is published for Anthropic's top_k, so nothing is refused;
        # the endpoint's own answer settles it. (`top_p` likewise has no BAND
        # -- its with-thinking window is a property of the pair, guarded where
        # thinking is resolved, and `sampling_band` deliberately says nothing
        # about it.)
        dec = resolved_decoding_params(
            "claude-sonnet-4-6", sampling={"top_k": 99999}, max_tokens=4096)
        assert dec["top_k"] == 99999

    def test_a_non_numeric_value_is_refused_where_a_band_exists(self):
        with pytest.raises(ValueError, match="not a number"):
            resolved_decoding_params(
                "claude-sonnet-4-6", sampling={"temperature": "hot"},
                max_tokens=4096)

    def test_a_rejected_param_is_dropped_before_the_band_is_read(self):
        # Opus 5 refuses temperature outright; the declared refusal drops it
        # (honest omission) and no band is consulted, however wild the value.
        dec = resolved_decoding_params(
            "claude-opus-5", sampling={"temperature": 9.9}, max_tokens=4096)
        assert "temperature" not in dec


class TestModelInfoErrors:
    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="unknown model"):
            model_info("totally-made-up-model-9000")

    def test_unknown_model_error_lists_known(self):
        with pytest.raises(ValueError, match="Known models"):
            model_info("nope")


class TestEveryPriceLivesInOnePlace:
    """A unit price appears in exactly one seam: `direktoro.prices`, where each
    rate is dated and carries the page it was read from. The registry states how
    to REACH a model and what it can do, and the arithmetic takes its rates as
    an argument, so a number without provenance has nowhere to sit.

    The guard is written as a PROPERTY OF NAMES rather than a list of approved
    ones. A price-shaped word anywhere on the public surface, on the `Model`
    record or in the registry module has to belong to the price seam, whatever
    it is called — which still works against a name nobody has invented yet."""

    # Words that appear in a name here only where money does. Matched
    # case-insensitively as substrings.
    PRICE_WORDS = ("price", "pricing", "rate", "cost", "usd", "dollar",
                   "tier", "discount", "billing", "invoice")

    # Public names carrying one of those words while holding no rate. Each says
    # what it is instead:
    #   - cost_from_rates, cost_from_usage: the arithmetic entry points, which
    #     take the rates as an ARGUMENT.
    #   - ProviderRateLimitError: a rate LIMIT is a ceiling on request
    #     frequency, and says nothing about what anything costs.
    EXEMPT = {"cost_from_rates", "cost_from_usage", "ProviderRateLimitError"}

    # The price seam itself: the dated table, its lookups, and the version stamp
    # that identifies which table priced a run. These are the names allowed to
    # be about money on the public surface, and every other price-shaped one
    # fails the check below.
    PRICE_SEAM = {"PRICES", "PRICES_VERSION", "PriceEntry", "price_for",
                  "is_priced", "price_age_days"}

    def _price_shaped(self, names):
        return sorted(name for name in names
                      if name not in self.EXEMPT
                      and any(word in name.lower()
                              for word in self.PRICE_WORDS))

    def test_the_model_record_declares_no_price_shaped_field(self):
        import dataclasses

        fields = [f.name for f in dataclasses.fields(Model)]
        assert self._price_shaped(fields) == [], (
            "Model has grown a price-shaped field. A rate belongs in "
            "direktoro.prices, where it carries its source and read date.")

    def test_no_registry_entry_carries_a_price_shaped_attribute(self):
        for model_id, entry in MODEL_REGISTRY.items():
            attributes = [name for name in vars(entry)
                          if not name.startswith("_")]
            assert self._price_shaped(attributes) == [], model_id

    def test_every_price_shaped_public_name_belongs_to_the_price_seam(self):
        import direktoro

        assert set(self._price_shaped(direktoro.__all__)) == self.PRICE_SEAM, (
            "a price-shaped name on the public surface belongs to neither the "
            "price table nor the arithmetic. Money is one seam; put it there.")

    def test_the_registry_module_holds_no_price_shaped_name(self):
        import direktoro.registry as registry

        public = [name for name in vars(registry) if not name.startswith("_")]
        assert self._price_shaped(public) == [], (
            "the registry has grown a price-shaped name. It states how to "
            "reach a model and what it can do; rates are direktoro.prices.")

    def test_the_arithmetic_cannot_answer_without_rates(self):
        # The positive half of the same property: rates are an argument, so
        # there is no default rate card for a total to be computed against.
        with pytest.raises(TypeError):
            cost_from_rates(input_tokens=10)
        with pytest.raises(TypeError):
            cost_from_usage(NormalisedUsage(input_tokens=10))


class TestAnthropicCapabilityBlock:
    """EVERY live Anthropic entry's decoding capabilities, verified against the
    published model reference on 2026-08-01.

    Mechanised rather than left in a comment: an entry that claims the wrong
    sampling surface makes a real call 400 after it has been queued, and a
    capability nothing re-checks is a claim, not a fact.

    The membership of the table is asserted rather than curated
    (`test_the_table_covers_every_live_entry`), so a new Anthropic entry fails
    here until it has been verified and added — the single most-used entry is
    exactly the one a hand-maintained list tends to omit."""

    # (model id, rejects the temperature parameter)
    VERIFIED = [
        ("claude-opus-5", True),
        ("claude-opus-4-8", True),
        ("claude-opus-4-7", True),
        ("claude-sonnet-5", True),
        ("claude-sonnet-4-6", False),
        ("claude-haiku-4-5-20251001", False),
    ]

    def test_the_table_covers_every_live_entry(self):
        live = sorted(model_id for model_id, info in MODEL_REGISTRY.items()
                      if info.provider == "anthropic" and not info.retired)
        assert sorted(row[0] for row in self.VERIFIED) == live

    @pytest.mark.parametrize("model_id,no_temp", VERIFIED)
    def test_sampling_refusal_is_declared(self, model_id, no_temp):
        rejected = model_info(model_id).rejects_sampling
        # The family that refuses them refuses all three, so the declaration is
        # the whole set or nothing.
        assert (rejected == frozenset(SAMPLING_PARAMS)) is no_temp

    @pytest.mark.parametrize("model_id,no_temp", VERIFIED)
    def test_the_declaration_reaches_the_resolved_decoding_params(
            self, model_id, no_temp):
        # The declaration is only worth recording if it actually removes the
        # param from what is sent (and from the fingerprint), so assert the
        # effect, not the flag.
        dec = resolved_decoding_params(model_id, sampling={"temperature": 0.0},
                                       max_tokens=4096)
        assert ("temperature" in dec) is not no_temp

    @pytest.mark.parametrize("model_id,no_temp", VERIFIED)
    def test_none_are_retired_or_routed(self, model_id, no_temp):
        info = model_info(model_id)
        assert info.retired is False
        assert info.route is None
        assert info.supports_images is True


class TestSonnet5Entry:
    """claude-sonnet-5: registered (unknown ids fail loud at startup, so nothing
    downstream can name it until it lands here), reachable at Anthropic, and it
    rejects temperature like the Opus 4.7+ family."""

    def test_is_known(self):
        assert is_known_model("claude-sonnet-5")

    def test_routes_to_anthropic_and_supports_vision(self):
        m = model_info("claude-sonnet-5")
        assert m.provider == "anthropic"
        assert m.base_url is None
        assert m.api_key_env == "ANTHROPIC_API_KEY"
        assert m.supports_images is True

    def test_rejects_temperature(self):
        # Sonnet 5's sampling params return a 400 (Claude model reference), so
        # the entry declares all three refused and the adapter never sends one.
        assert model_info("claude-sonnet-5").rejects_sampling == frozenset(
            SAMPLING_PARAMS)
        dec = resolved_decoding_params("claude-sonnet-5", sampling={"temperature": 0.0},
                                       max_tokens=4096)
        assert "temperature" not in dec

    def test_sonnet_4_6_still_takes_temperature(self):
        # The neighbouring generation declares no refusal, so the resolver
        # sends what it is given — asserted through the effect, since the flag
        # alone would not prove the temperature reaches the wire.
        assert model_info("claude-sonnet-4-6").rejects_sampling == frozenset()
        dec = resolved_decoding_params(
            "claude-sonnet-4-6", sampling={"temperature": 0.0}, max_tokens=4096)
        assert dec["temperature"] == 0.0


class TestOpus5Entry:
    """claude-opus-5: Anthropic's current Opus. Direct, vision-capable, and it
    rejects temperature like the rest of the 4.7+ family."""

    def test_is_known(self):
        assert is_known_model("claude-opus-5")

    def test_routes_to_anthropic_and_supports_vision(self):
        m = model_info("claude-opus-5")
        assert m.provider == "anthropic"
        assert m.base_url is None
        assert m.api_key_env == "ANTHROPIC_API_KEY"
        assert m.supports_images is True
        assert m.route is None          # direct, not gateway-served
        assert m.retired is False

    def test_rejects_temperature(self):
        assert model_info("claude-opus-5").rejects_sampling == frozenset(
            SAMPLING_PARAMS)
        dec = resolved_decoding_params("claude-opus-5", sampling={"temperature": 0.0},
                                       max_tokens=4096)
        assert "temperature" not in dec


class TestModelInfo:
    def test_anthropic_metadata(self):
        m = model_info("claude-opus-4-8")
        assert m.provider == "anthropic"
        assert m.base_url is None
        assert m.api_key_env == "ANTHROPIC_API_KEY"
        assert m.rejects_sampling == frozenset(SAMPLING_PARAMS)

    def test_openai_metadata(self):
        m = model_info("gpt-5.6-sol")
        assert m.provider == "openai"
        assert m.base_url == "https://api.openai.com/v1"
        assert m.api_key_env == "OPENAI_API_KEY"
        # Its documentation establishes `temperature` and nothing further.
        assert m.rejects_sampling == frozenset({"temperature"})
        assert m.quirks["reasoning_effort"] == "medium"

    def test_routed_glm_metadata(self):
        # Routed entries speak the OpenRouter OpenAI-compat Chat Completions
        # wire, keyed by the OpenRouter base URL and OPENROUTER_API_KEY, and
        # carry a Route.
        m = model_info("z-ai/glm-4.6v")
        assert m.provider == "openrouter"
        assert m.base_url == "https://openrouter.ai/api/v1"
        assert m.api_key_env == "OPENROUTER_API_KEY"
        assert m.wire_api == "chat_completions"
        assert m.supports_images is True
        assert m.route is not None
        assert m.route.gateway == "openrouter"
        assert m.route.upstream == ("z-ai",)
        assert m.route.quantizations == ("fp8",)

    def test_routed_qwen_metadata(self):
        m = model_info("qwen/qwen3-vl-235b-a22b-instruct")
        assert m.provider == "openrouter"
        assert m.base_url == "https://openrouter.ai/api/v1"
        assert m.api_key_env == "OPENROUTER_API_KEY"
        assert m.wire_api == "chat_completions"
        assert m.supports_images is True
        assert m.route is not None
        # Pinned to Venice + Parasail at fp8: the first-party upstream (Alibaba)
        # offers no ZDR through OpenRouter (confirmed live), so under this
        # entry's privacy pin the model is served by ZDR-capable third-party
        # hosts instead.
        assert m.route.upstream == ("venice", "parasail")
        assert m.route.quantizations == ("fp8",)


class TestRoutedFrontierEntries:
    """The three routed frontier entries: xiaomi/mimo-v2.5 (Parasail + Venice,
    fp8) and the two Gemini Flash entries (both on Vertex, but at DIFFERENT
    service tiers — 3.6 flex, 3.7 standard). All are gateway-served and
    vision-capable; they differ on forced tool_choice, and both Gemini entries
    additionally take no sampling params."""

    def test_all_are_known_and_routed(self):
        for m in ("xiaomi/mimo-v2.5", "google/gemini-3.6-flash",
                  "google/gemini-3.7-flash"):
            info = model_info(m)
            assert info.provider == "openrouter"
            assert info.base_url == "https://openrouter.ai/api/v1"
            assert info.api_key_env == "OPENROUTER_API_KEY"
            assert info.wire_api == "chat_completions"
            assert info.supports_images is True
            assert info.route is not None

    def test_gemini_35_is_absent(self):
        # 3.6 supersedes 3.5, and 3.5 has never been a registry key, so no
        # stored run can name it. That is the one case in which deleting an
        # entry beats flagging it retired (see the registry's DELETING AN ENTRY
        # note): with no provenance to keep resolvable, the slug is simply
        # unknown.
        assert not is_known_model("google/gemini-3.5-flash")
        assert "google/gemini-3.5-flash" not in MODEL_REGISTRY

    def test_mimo_pin_two_hosts_fp8_and_auto_tool(self):
        info = model_info("xiaomi/mimo-v2.5")
        # Two hosts pinned against upstream churn, Parasail first; declared fp8.
        assert info.route.upstream == ("parasail", "venice")
        assert info.route.quantizations == ("fp8",)
        # Parasail 404s a forced named tool_choice for this slug (like the Z.AI
        # GLM endpoints); to keep the pinned set uniform the entry runs "auto"
        # and a caller retries.
        assert info.forced_tool_choice is False
        # MiMo declares no sampling refusal, so it is sent what it is given.
        assert info.rejects_sampling == frozenset()

    def test_gemini_pin_vertex_flex_tag_forced_tool_and_no_sampling(self):
        info = model_info("google/gemini-3.6-flash")
        # The Vertex flex tier is pinned by the endpoint tag in provider.order.
        assert info.route.upstream == ("google-vertex/global/flex",)
        # Proprietary: no quant tags, so no quant filter (would wrongly exclude).
        assert info.route.quantizations == ()
        # Forced named tool_choice verified live at the flex endpoint 2026-07-24.
        assert info.forced_tool_choice is True
        # 3.6's Vertex endpoints list no sampling control at all — temperature,
        # top_p and top_k are equally absent from the supported_parameters read
        # of 2026-07-24 — so the entry names all three.
        assert info.rejects_sampling == frozenset(
            {"temperature", "top_p", "top_k"})

    def test_gemini_37_pins_vertex_standard_not_flex(self):
        info = model_info("google/gemini-3.7-flash")
        # Same gateway, same key, same wire and the same Vertex provider as 3.6,
        # but a DIFFERENT service tier, and the difference is the point. 3.7
        # lists the identical three Vertex tiers (global / flex / priority) plus
        # three Google AI Studio ones; the bare-region `google-vertex/global` tag
        # is the DEFAULT (standard, on-demand) endpoint. It was flex until
        # 2026-08-17, when a 250-call consumer run against the flex pin failed 56
        # calls (32 x HTTP 524/504 upstream origin timeouts, 24 x 429) evenly
        # across ~100 minutes — sustained capacity-shedding, not a burst — and
        # the owner ruled the standard tier with the ~2x price accepted.
        # Asserted rather than described so a hand tidying this row back into
        # line with its 3.6 neighbour has to bring that ruling with it.
        assert info.route.upstream == ("google-vertex/global",)
        assert "flex" not in info.route.upstream[0]
        # Not the same pin as 3.6, deliberately: the two rows disagree about the
        # tier and neither is the other's typo.
        assert (info.route.upstream
                != model_info("google/gemini-3.6-flash").route.upstream)
        # Proprietary, quantization reported "unknown" on every endpoint, so no
        # quant filter — one would exclude the only host there is.
        assert info.route.quantizations == ()
        # Single-host pin, so the fallback discipline is what stops a silent
        # reroute: fail rather than land somewhere unvetted.
        assert info.route.allow_fallbacks is False
        assert info.route.zdr is True
        assert info.route.data_collection == "deny"
        # Forced named tool_choice is stated on DOCUMENTATION (Google's
        # function-calling reference and the gateway's supported_parameters,
        # read 2026-08-17), not on a probe of this slug — which the entry says
        # in place. Pinned here so flipping it later has to bring the live call
        # that justifies the flip.
        assert info.forced_tool_choice is True
        # The pinned Vertex endpoints list no sampling control at all (read
        # 2026-08-17). The SAME slug's Google AI Studio endpoints do list
        # temperature and top_p, so this refusal describes the pin, not the
        # model — re-pinning would have to re-read the field.
        assert info.rejects_sampling == frozenset(
            {"temperature", "top_p", "top_k"})
        # Figure crops are the reason this entry exists.
        assert info.supports_images is True

    def test_gemini_resolver_drops_temperature_mimo_keeps_it(self):
        # The resolution seam: resolved_decoding_params (the single source of
        # truth for wire AND fingerprint) omits temperature for the no-sampling
        # model but keeps it for a sampling-capable routed peer.
        gem = resolved_decoding_params(
            "google/gemini-3.6-flash", sampling={"temperature": 0.0}, max_tokens=4096)
        assert gem == {"max_tokens": 4096}
        mimo = resolved_decoding_params(
            "xiaomi/mimo-v2.5", sampling={"temperature": 0.0}, max_tokens=4096)
        assert mimo == {"max_tokens": 4096, "temperature": 0.0}


# ---------------------------------------------------------------------------
# cost_from_rates: arithmetic over the caller's own rates
# ---------------------------------------------------------------------------

# A rate card in the shape a caller passes: USD per MILLION tokens, keyed by
# counter. Round numbers so every expected total below is arithmetic anyone can
# check by eye. The two cache-write rates differ, which is the whole reason
# there are two counters: 3.75 is 1.25x the 3.0 input rate (a 5-minute write)
# and 6.0 is 2x it (a 1-hour write).
RATES = {"input": 3.0, "output": 15.0, "cache_read": 0.30,
         "cache_write": 3.75, "cache_write_1h": 6.0}


class TestCostFromRates:
    def test_input_and_output(self):
        assert cost_from_rates(rates=RATES, input_tokens=1_000_000,
                               output_tokens=1_000_000) == \
            pytest.approx(3.0 + 15.0)

    def test_every_counter_sums(self):
        cost = cost_from_rates(
            rates=RATES, input_tokens=1_000_000, output_tokens=1_000_000,
            cache_read_tokens=1_000_000, cache_write_tokens=1_000_000,
            cache_write_1h_tokens=1_000_000)
        assert cost == pytest.approx(3.0 + 15.0 + 0.30 + 3.75 + 6.0)

    def test_the_two_cache_write_ttls_are_priced_at_their_own_rates(self):
        # The reason cache writes are two counters rather than one. Anthropic
        # bills a 5-minute write at 1.25x the base input rate and a 1-hour write
        # at 2x, so a million tokens of each is 3.75 + 6.00 and never 2 x either.
        cost = cost_from_rates(rates=RATES, cache_write_tokens=1_000_000,
                               cache_write_1h_tokens=1_000_000)
        assert cost == pytest.approx(3.75 + 6.0)
        assert cost != pytest.approx(2 * 3.75)
        assert cost != pytest.approx(2 * 6.0)

    def test_partial_token_counts_scale_per_million(self):
        cost = cost_from_rates(rates=RATES, input_tokens=100, output_tokens=20)
        assert cost == pytest.approx(100 / 1e6 * 3.0 + 20 / 1e6 * 15.0)

    def test_every_counter_defaults_to_zero(self):
        assert cost_from_rates(rates=RATES) == 0.0

    def test_an_empty_rate_card_costs_zero_usage_at_zero(self):
        # Nothing was used, so nothing needs a rate.
        assert cost_from_rates(rates={}) == 0.0

    def test_extra_keys_in_the_rate_card_are_ignored(self):
        # A caller may hand over its whole rate card without filtering it.
        rates = dict(RATES, audio=100.0, notes="from the July invoice")
        assert cost_from_rates(rates=rates, input_tokens=1_000_000) == \
            pytest.approx(3.0)

    def test_the_documented_counter_names_are_the_ones_priced(self):
        assert COUNTERS == ("input", "output", "cache_read", "cache_write",
                            "cache_write_1h")

    def test_it_needs_no_model_id(self):
        # The property that makes this un-rottable: the arithmetic knows nothing
        # about any model, provider or registry, so no provider price change can
        # make it wrong. Passing a model id is not even possible.
        import inspect

        parameters = inspect.signature(cost_from_rates).parameters
        assert set(parameters) == {
            "rates", "input_tokens", "output_tokens", "cache_read_tokens",
            "cache_write_tokens", "cache_write_1h_tokens"}
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY
                   for p in parameters.values())

    def test_every_counter_has_a_keyword_argument_of_its_own(self):
        # The counter list and the signature are one thing said twice, so they
        # are held together: a counter added to COUNTERS with no argument to
        # supply it would be permanently zero, and an argument with no counter
        # would be silently ignored.
        import inspect

        parameters = set(inspect.signature(cost_from_rates).parameters)
        assert {f"{counter}_tokens" for counter in COUNTERS} == \
            parameters - {"rates"}


class TestAMissingRateIsLoud:
    """The whole point of taking rates as an argument: under-reporting must be
    impossible.

    Tokens in a counter were genuinely billed. Costing them at zero because no
    rate was passed produces a total that is quietly too low — and a cost that
    is wrong in the cheap direction is the one nobody questions. So a non-zero
    counter with no rate raises, naming the counter."""

    @pytest.mark.parametrize("counter,kwargs", [
        ("input", {"input_tokens": 10}),
        ("output", {"output_tokens": 10}),
        ("cache_read", {"cache_read_tokens": 10}),
        ("cache_write", {"cache_write_tokens": 10}),
        ("cache_write_1h", {"cache_write_1h_tokens": 10}),
    ])
    def test_a_non_zero_counter_with_no_rate_raises(self, counter, kwargs):
        with pytest.raises(ValueError) as excinfo:
            cost_from_rates(rates={}, **kwargs)
        message = str(excinfo.value)
        assert counter in message          # names the counter that was unpriced
        assert "10" in message             # and how many tokens went unpriced

    def test_the_other_counters_being_priced_does_not_excuse_it(self):
        # The realistic shape of the bug: a caller passes input/output rates,
        # the provider starts reporting cached reads, and those tokens would
        # ride free forever.
        with pytest.raises(ValueError, match="cache_read"):
            cost_from_rates(rates={"input": 3.0, "output": 15.0},
                            input_tokens=1_000, output_tokens=100,
                            cache_read_tokens=500)

    def test_a_zero_counter_needs_no_rate(self):
        # Nothing to price, so nothing to complain about: a caller passes only
        # the rates its usage actually calls for.
        assert cost_from_rates(rates={"input": 3.0}, input_tokens=1_000_000,
                               output_tokens=0, cache_read_tokens=0,
                               cache_write_tokens=0,
                               cache_write_1h_tokens=0) == pytest.approx(3.0)

    def test_the_guard_cannot_fire_on_a_counter_that_was_never_passed(self):
        # The shape of what this guard can and cannot do, pinned so the gap is
        # documented rather than assumed away. An OMITTED counter defaults to
        # zero, and zero needs no rate, so a call site that forgets to thread
        # one through gets a confident, silent, too-low total. That is why
        # `cost_from_usage` exists: it enumerates the counters instead of
        # leaving each call site to name them.
        assert cost_from_rates(rates={"input": 3.0}, input_tokens=1_000_000) \
            == pytest.approx(3.0)
        # The very same usage, with the cache counters actually supplied, is a
        # different number — and would have raised had their rates been absent.
        assert cost_from_rates(
            rates=RATES, input_tokens=1_000_000, cache_read_tokens=1_000_000,
            cache_write_tokens=1_000_000) == pytest.approx(3.0 + 0.30 + 3.75)

    def test_an_explicit_zero_rate_is_honoured_not_treated_as_missing(self):
        # A provider that genuinely does not charge for a counter is a rate of
        # zero, which is a statement. Only an ABSENT rate is the failure.
        assert cost_from_rates(rates={"input": 3.0, "cache_write": 0.0},
                               input_tokens=1_000_000,
                               cache_write_tokens=1_000_000) == \
            pytest.approx(3.0)

    def test_the_message_says_how_to_fix_it(self):
        with pytest.raises(ValueError) as excinfo:
            cost_from_rates(rates={}, output_tokens=42)
        message = str(excinfo.value)
        assert "rates=" in message
        assert "per million" in message


class TestMalformedInputIsRefused:
    """A negative anywhere in this sum moves the total DOWN, which is the one
    direction that must never happen by accident."""

    @pytest.mark.parametrize("counter", COUNTERS)
    def test_a_negative_rate_raises(self, counter):
        rates = dict(RATES, **{counter: -1.0})
        with pytest.raises(ValueError, match="negative"):
            cost_from_rates(rates=rates, **{f"{counter}_tokens": 1_000})

    @pytest.mark.parametrize("counter", COUNTERS)
    def test_a_negative_token_count_raises(self, counter):
        with pytest.raises(ValueError) as excinfo:
            cost_from_rates(rates=RATES, **{f"{counter}_tokens": -1})
        message = str(excinfo.value)
        assert counter in message
        assert "negative" in message

    def test_a_negative_count_is_refused_before_a_missing_rate_hides_it(self):
        # Order matters: a negative count with no rate must still report the
        # negative, not be swallowed by the missing-rate path.
        with pytest.raises(ValueError, match="negative"):
            cost_from_rates(rates={}, input_tokens=-5)


# ---------------------------------------------------------------------------
# cost_from_usage: the whole of a usage record, priced, with nothing dropped
# ---------------------------------------------------------------------------

def _usage(**counts):
    """A `NormalisedUsage` with the named counters set and every other zero."""
    return NormalisedUsage(**counts)


class TestCostFromUsage:
    """Pricing a whole `NormalisedUsage`, which is the shape a caller actually
    has after a call.

    The mapping between a usage record's field names and a rate card's counter
    names is where a counter goes missing — they are genuinely different
    vocabularies — and `cost_from_rates` cannot catch that, because an omitted
    argument is zero and zero asks for no rate. Enumerating the counters here
    once is what closes it."""

    FULL = dict(input_tokens=1_000_000, output_tokens=1_000_000,
                cache_read_input_tokens=1_000_000,
                cache_creation_input_tokens=1_000_000,
                cache_creation_5m_input_tokens=400_000,
                cache_creation_1h_input_tokens=600_000)

    def test_every_counter_on_the_record_is_priced(self):
        cost = cost_from_usage(_usage(**self.FULL), rates=RATES)
        assert cost == pytest.approx(
            3.0                            # 1M full-rate input
            + 15.0                         # 1M output
            + 0.30                         # 1M cached reads
            + 0.4 * 3.75                   # 400K 5-minute cache writes
            + 0.6 * 6.0)                   # 600K 1-hour cache writes

    def test_it_agrees_with_naming_every_counter_by_hand(self):
        # Same arithmetic, reached the long way. The value is not that the two
        # agree but that only one of them can leave a counter out.
        usage = _usage(**self.FULL)
        assert cost_from_usage(usage, rates=RATES) == pytest.approx(
            cost_from_rates(
                rates=RATES,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_input_tokens,
                cache_write_tokens=usage.cache_creation_5m_input_tokens,
                cache_write_1h_tokens=usage.cache_creation_1h_input_tokens))

    def test_the_counter_a_hand_written_mapping_forgets_is_not_dropped(self):
        # The mistake this exists to catch, made concrete: a call site that
        # threads only input and output through prices a cached read at nothing
        # and never hears about it. Through the usage record it is priced.
        usage = _usage(input_tokens=1_000_000,
                       cache_read_input_tokens=1_000_000)
        forgotten = cost_from_rates(rates=RATES,
                                    input_tokens=usage.input_tokens,
                                    output_tokens=usage.output_tokens)
        assert forgotten == pytest.approx(3.0)          # silently too low
        assert cost_from_usage(usage, rates=RATES) == pytest.approx(3.0 + 0.30)

    def test_a_usage_with_no_cache_activity_prices_from_two_rates(self):
        usage = _usage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost_from_usage(
            usage, rates={"input": 3.0, "output": 15.0}) == \
            pytest.approx(18.0)

    def test_an_empty_usage_costs_nothing_and_needs_no_rates(self):
        assert cost_from_usage(_usage(), rates={}) == 0.0

    def test_a_missing_rate_is_still_loud_through_this_path(self):
        # cost_from_usage adds a counter check; it does not soften the rate
        # check underneath it.
        with pytest.raises(ValueError, match="cache_read"):
            cost_from_usage(
                _usage(input_tokens=1_000, cache_read_input_tokens=500),
                rates={"input": 3.0, "output": 15.0})

    def test_a_record_missing_a_counter_field_fails_loudly(self):
        # Reading by attribute with no fallback: a record that does not carry a
        # counter is a broken record, not a record whose counter is zero.
        class _Partial:
            input_tokens = 10
            output_tokens = 0

        with pytest.raises(AttributeError):
            cost_from_usage(_Partial(), rates=RATES)


class TestCacheCreationIsNeverLeftUnattributed:
    """A cache-write total that the per-TTL split does not account for cannot be
    priced, and this refuses to pretend otherwise.

    A 5-minute write bills at 1.25x the base input rate and a 1-hour write at
    2x, so the unsplit total is not a priceable quantity: costing the remainder
    at zero under-reports, and assigning it to a TTL is a guess that comes out
    of the function looking exactly like a measurement. Both are refused."""

    def test_a_split_that_sums_to_the_total_is_accepted(self):
        usage = _usage(cache_creation_input_tokens=1_000_000,
                       cache_creation_5m_input_tokens=250_000,
                       cache_creation_1h_input_tokens=750_000)
        assert cost_from_usage(usage, rates=RATES) == pytest.approx(
            0.25 * 3.75 + 0.75 * 6.0)

    def test_a_total_with_no_split_at_all_raises(self):
        # The realistic case: a response reports the cache-write total but not
        # the nested per-TTL counts. There is no answer to be had without
        # knowing which TTL was asked for, so none is invented.
        usage = _usage(cache_creation_input_tokens=1_000_000)
        with pytest.raises(ValueError) as excinfo:
            cost_from_usage(usage, rates=RATES)
        message = str(excinfo.value)
        assert "1000000" in message
        assert "0" in message

    def test_a_partial_split_raises_not_prices_the_remainder_free(self):
        usage = _usage(cache_creation_input_tokens=1_000_000,
                       cache_creation_5m_input_tokens=400_000)
        with pytest.raises(ValueError, match="not fully attributed"):
            cost_from_usage(usage, rates=RATES)

    def test_a_split_exceeding_the_total_raises_too(self):
        # Over-attribution is just as much a broken record as under-, and it
        # moves the total the other way.
        usage = _usage(cache_creation_input_tokens=1_000,
                       cache_creation_5m_input_tokens=1_000,
                       cache_creation_1h_input_tokens=1_000)
        with pytest.raises(ValueError, match="not fully attributed"):
            cost_from_usage(usage, rates=RATES)

    def test_no_cache_writes_at_all_is_not_a_missing_split(self):
        # Where a provider reports no cache writes, all three counts are zero,
        # the sums agree, and there is nothing to attribute.
        usage = _usage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert usage.cache_creation_input_tokens == 0
        assert cost_from_usage(usage, rates=RATES) == pytest.approx(18.0)

    def test_the_message_names_the_total_and_what_the_split_accounts_for(self):
        usage = _usage(cache_creation_input_tokens=900,
                       cache_creation_5m_input_tokens=100,
                       cache_creation_1h_input_tokens=200)
        with pytest.raises(ValueError) as excinfo:
            cost_from_usage(usage, rates=RATES)
        message = str(excinfo.value)
        assert "900" in message          # what the record says was written
        assert "300" in message          # what the split accounts for
        assert "100" in message and "200" in message   # and at which TTLs

    def test_the_refusal_beats_the_zero_it_would_otherwise_produce(self):
        # Stated as the property rather than the mechanism: the unattributed
        # tokens would cost nothing if this were allowed through, and a total
        # that is quietly too low is the one nobody questions.
        usage = _usage(cache_creation_input_tokens=1_000_000)
        priced_if_allowed = cost_from_rates(
            rates=RATES, cache_write_tokens=0, cache_write_1h_tokens=0)
        assert priced_if_allowed == 0.0
        with pytest.raises(ValueError):
            cost_from_usage(usage, rates=RATES)
