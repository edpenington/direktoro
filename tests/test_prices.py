"""Tests for the dated price table (direktoro.prices).

An entry is a reading of a vendor's published rate, and these hold the table to
what that makes it: every entry complete and non-negative, every reading dated
and sourced, every key a model this package can actually reach, and every
mapping `as_rates()` produces one that `cost_from_rates` prices without the
caller rekeying anything.

The table covers the direct models. A routed call carries the gateway's own
charge on the response, so a routed id has no entry and `price_for` answers
None for it.
"""

from datetime import date, timedelta

import pytest

from direktoro.cost import COUNTERS, cost_from_rates
from direktoro.prices import (
    PRICES, PRICES_VERSION, PriceEntry, is_priced, price_age_days, price_for)
from direktoro.registry import MODEL_REGISTRY, is_known_model


# The rate fields, paired with the `as_rates()` counter each one feeds and the
# `cost_from_rates` token argument that counter is billed under. Enumerated once
# so the tests below cover all four rather than whichever ones they name.
RATES = (
    ("input_per_1m", "input", "input_tokens"),
    ("output_per_1m", "output", "output_tokens"),
    ("cache_read_per_1m", "cache_read", "cache_read_tokens"),
    ("cache_write_per_1m", "cache_write", "cache_write_tokens"),
)


class TestEveryEntryIsAWholeReading:
    """A partial entry is the dangerous kind: a rate left off prices its counter
    at nothing the moment a call reports tokens in it, and an undated or
    unsourced one cannot be checked against the page it came from."""

    def test_every_entry_carries_all_four_rates_non_negative(self):
        for model_id, entry in PRICES.items():
            for field, _, _ in RATES:
                rate = getattr(entry, field)
                assert isinstance(rate, (int, float)), (model_id, field)
                assert rate >= 0, (model_id, field)

    def test_every_as_of_is_an_iso_date(self):
        for model_id, entry in PRICES.items():
            assert date.fromisoformat(entry.as_of), model_id

    def test_every_source_is_an_https_url(self):
        for model_id, entry in PRICES.items():
            assert entry.source.startswith("https://"), model_id

    def test_the_table_version_is_a_bumpable_integer(self):
        # What a consumer records to say which table priced a run.
        assert isinstance(PRICES_VERSION, int)
        assert PRICES_VERSION >= 1


class TestWhatTheTableCovers:
    def test_every_key_is_a_registered_model_id(self):
        for model_id in PRICES:
            assert is_known_model(model_id), (
                f"{model_id!r} is priced but has no registry entry, so nothing "
                f"can reach the model this rate describes.")

    def test_no_routed_model_is_priced(self):
        for model_id, info in MODEL_REGISTRY.items():
            if info.route is None:
                continue
            assert model_id not in PRICES, (
                f"{model_id!r} is routed: its cost comes back from the gateway "
                f"on the response (NormalisedResponse.reported_cost), so a "
                f"table entry for it would never be read.")

    def test_every_startable_direct_model_is_priced(self):
        # Asserted rather than curated: a new direct entry fails here until its
        # rate has been read off the vendor's page and dated. Retired direct ids
        # are priced only while the vendor still publishes a rate for them.
        unpriced = sorted(
            model_id for model_id, info in MODEL_REGISTRY.items()
            if info.route is None and not info.retired
            and model_id not in PRICES)
        assert unpriced == [], (
            f"direct models with no price entry: {unpriced}. Read the vendor's "
            f"published rate, add the entry with its source and read date, and "
            f"bump PRICES_VERSION.")


class TestLookups:
    def test_price_for_returns_the_entry_of_a_priced_model(self):
        entry = price_for("claude-opus-5")
        assert isinstance(entry, PriceEntry)
        assert entry.input_per_1m == 5.00
        assert entry.output_per_1m == 25.00

    def test_price_for_returns_none_for_an_unknown_id(self):
        assert price_for("totally-made-up-model-9000") is None

    def test_price_for_returns_none_for_a_routed_id(self):
        routed = [model_id for model_id, info in MODEL_REGISTRY.items()
                  if info.route is not None]
        assert routed, "no routed entry to check against"
        for model_id in routed:
            assert price_for(model_id) is None, model_id

    def test_is_priced_agrees_with_the_table(self):
        for model_id in PRICES:
            assert is_priced(model_id) is True
        assert is_priced("z-ai/glm-5v-turbo") is False
        assert is_priced("totally-made-up-model-9000") is False


class TestPriceAgeDays:
    MODEL = "claude-opus-5"

    def _read_date(self):
        return date.fromisoformat(PRICES[self.MODEL].as_of)

    def test_the_age_is_zero_on_the_day_the_page_was_read(self):
        assert price_age_days(self.MODEL, self._read_date()) == 0

    def test_the_age_grows_one_per_day(self):
        read = self._read_date()
        ages = [price_age_days(self.MODEL, read + timedelta(days=n))
                for n in range(6)]
        assert ages == [0, 1, 2, 3, 4, 5]
        assert ages == sorted(ages)

    def test_a_date_before_the_reading_counts_backwards(self):
        # The count is `today` minus the reading, in both directions, so a
        # caller checking an age against a run's own date gets an arithmetic
        # answer rather than a floor at zero.
        read = self._read_date()
        assert price_age_days(self.MODEL, read - timedelta(days=3)) == -3

    def test_it_reads_no_clock_of_its_own(self):
        # Same arguments, same answer, whenever the suite runs: the date is the
        # caller's, so a recorded age recomputes exactly.
        today = date(2026, 9, 1)
        assert price_age_days(self.MODEL, today) == price_age_days(
            self.MODEL, today)
        assert price_age_days(self.MODEL, today) == (
            today - self._read_date()).days

    def test_an_unpriced_model_has_no_age_to_report(self):
        with pytest.raises(KeyError, match="no price entry"):
            price_age_days("z-ai/glm-5v-turbo", date(2026, 9, 1))
        with pytest.raises(KeyError, match="no price entry"):
            price_age_days("totally-made-up-model-9000", date(2026, 9, 1))


class TestAsRatesFeedsTheArithmetic:
    """`as_rates()` exists so nothing between the table and the arithmetic has
    to rekey a counter, and a rekeying is exactly where one goes missing: a
    counter `cost_from_rates` has tokens for and no rate for raises, and a
    counter nobody passes costs nothing at all."""

    def test_the_keys_are_the_counters_cost_from_rates_prices(self):
        expected = {counter for _, counter, _ in RATES}
        for model_id, entry in PRICES.items():
            assert set(entry.as_rates()) == expected, model_id
            assert set(entry.as_rates()) <= set(COUNTERS), model_id

    def test_the_values_are_the_entrys_own_rates(self):
        for model_id, entry in PRICES.items():
            rates = entry.as_rates()
            for field, counter, _ in RATES:
                assert rates[counter] == getattr(entry, field), (
                    model_id, counter)

    def test_the_mapping_prices_a_call_without_rekeying(self):
        entry = PRICES["claude-opus-5"]
        total = cost_from_rates(
            rates=entry.as_rates(),
            input_tokens=12_000, output_tokens=800,
            cache_read_tokens=40_000, cache_write_tokens=2_000)
        expected = (12_000 * 5.00 + 800 * 25.00 + 40_000 * 0.50
                    + 2_000 * 6.25) / 1_000_000
        assert total == pytest.approx(expected)

    def test_each_counter_is_priced_at_the_field_it_came_from(self):
        # One million tokens in one counter and nothing in the others: the
        # total IS that counter's rate, so a mapping that crossed two fields
        # fails here rather than costing a run at the wrong one.
        entry = PRICES["claude-opus-5"]
        rates = entry.as_rates()
        for field, _, tokens_argument in RATES:
            total = cost_from_rates(rates=rates,
                                    **{tokens_argument: 1_000_000})
            assert total == pytest.approx(getattr(entry, field)), field

    def test_a_one_hour_cache_write_needs_a_rate_the_caller_adds(self):
        # These are the 5-minute-TTL write rates, so the 1-hour counter is
        # unpriced by the mapping and pricing tokens in it raises rather than
        # billing them at the 5-minute rate.
        entry = PRICES["claude-opus-5"]
        assert "cache_write_1h" not in entry.as_rates()
        with pytest.raises(ValueError, match="cache_write_1h"):
            cost_from_rates(rates=entry.as_rates(),
                            cache_write_1h_tokens=1_000)
