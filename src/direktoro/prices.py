"""Dated price table: what each vendor publishes, when it was read, from where.

Every entry records one model's published rates in USD per MILLION tokens
together with the date the vendor's page was read (`as_of`) and the page itself
(`source`), so a rate can be traced back to the document it came from and its
age measured (`price_age_days`). `PRICES_VERSION` names the table's data as a
whole, for a consumer recording which table priced a run.

The table covers DIRECT models. A routed call comes back carrying the gateway's
own charge (`NormalisedResponse.reported_cost`), which is the figure a routed
run records, so a routed entry here is a number nothing would ever read.

`cache_write_per_1m` is the 5-minute-TTL cache write, the API default. The
1-hour TTL is a different price, and `cost_from_rates` prices it under a
separate `cache_write_1h` counter, from rates a caller writing at that TTL
supplies.

Long-context pricing bands are not modelled: these are the standard-tier rates.
A caller whose requests cross a vendor's long-context threshold supplies the
banded rates itself.

`PriceEntry.as_rates()` hands an entry straight to
`direktoro.cost.cost_from_rates`.
"""

from dataclasses import dataclass
from datetime import date


# Identifies the DATA below, so a consumer can record which table priced a run
# and tell two runs apart when a rate moved between them. Bump it on any change
# to an entry — a rate, a date, a source, an addition, a removal. It is
# independent of the package version: a release that leaves the table alone
# leaves this alone.
PRICES_VERSION = 1

# The read date and vendor page behind every entry below. One of each per
# vendor today, so they are named once here and carried into the entries rather
# than retyped on every row.
_READ = "2026-08-07"
_ANTHROPIC_PRICING = "https://platform.claude.com/docs/en/about-claude/pricing"
_OPENAI_PRICING = "https://developers.openai.com/api/docs/pricing"


@dataclass(frozen=True)
class PriceEntry:
    """One model's published rates, in USD per MILLION tokens, with its source.

    The four rates are the counters a call is billed under: full-rate
    (cache-miss) input, output, cached reads, and 5-minute-TTL cache writes.
    `as_of` is the ISO date the `source` page was read, and the rates are what
    that page stated on that date.
    """

    input_per_1m: float
    output_per_1m: float
    cache_read_per_1m: float
    cache_write_per_1m: float
    as_of: str
    source: str

    def as_rates(self):
        """This entry as a `cost_from_rates` rates mapping, keyed by counter.

        Carries no `cache_write_1h` rate — these are the 5-minute-TTL write
        rates — so a call reporting 1-hour cache writes needs that rate added
        to the mapping (`cost_from_rates` raises for a counter it was given
        tokens but no rate for).
        """
        return {
            "input": self.input_per_1m,
            "output": self.output_per_1m,
            "cache_read": self.cache_read_per_1m,
            "cache_write": self.cache_write_per_1m,
        }


# Keys are registry model ids (`direktoro.registry.MODEL_REGISTRY`).
#
# Each entry's evidence is in its own fields: `source` is the page, `as_of` the
# day it was read. A comment beside an entry carries only what the fields
# cannot — a retirement note, a scheduled change.
#
# NOT PRICED HERE:
#   - `claude-3-5-sonnet-20241022`: retired, and the vendor publishes no rate
#     for it to record.
#   - the five routed (OpenRouter) entries: a routed call is priced from the
#     gateway's reported charge on the response.
PRICES = {
    # ---- Anthropic ----------------------------------------------------------
    # `cache_read_per_1m` is the page's "Cache Hits & Refreshes" column;
    # `cache_write_per_1m` is its "5m Cache Writes" column.
    "claude-opus-5": PriceEntry(
        5.00, 25.00, 0.50, 6.25, as_of=_READ, source=_ANTHROPIC_PRICING),
    "claude-opus-4-8": PriceEntry(
        5.00, 25.00, 0.50, 6.25, as_of=_READ, source=_ANTHROPIC_PRICING),
    "claude-opus-4-7": PriceEntry(
        5.00, 25.00, 0.50, 6.25, as_of=_READ, source=_ANTHROPIC_PRICING),
    # Introductory pricing through 2026-08-31; from 2026-09-01 the published
    # rate is 3.00 / 15.00 / 0.30 / 3.75 — update this entry and bump
    # PRICES_VERSION then.
    "claude-sonnet-5": PriceEntry(
        2.00, 10.00, 0.20, 2.50, as_of=_READ, source=_ANTHROPIC_PRICING),
    "claude-sonnet-4-6": PriceEntry(
        3.00, 15.00, 0.30, 3.75, as_of=_READ, source=_ANTHROPIC_PRICING),
    # Page row "Claude Haiku 4.5".
    "claude-haiku-4-5-20251001": PriceEntry(
        1.00, 5.00, 0.10, 1.25, as_of=_READ, source=_ANTHROPIC_PRICING),
    # Retired on the Claude API (still served on Bedrock and Google Cloud); the
    # vendor still publishes the rate.
    "claude-sonnet-4-20250514": PriceEntry(
        3.00, 15.00, 0.30, 3.75, as_of=_READ, source=_ANTHROPIC_PRICING),
    # Retired on the Claude API (still served on Google Cloud); the vendor still
    # publishes the rate.
    "claude-opus-4-20250514": PriceEntry(
        15.00, 75.00, 1.50, 18.75, as_of=_READ, source=_ANTHROPIC_PRICING),

    # ---- OpenAI -------------------------------------------------------------
    # Standard tier. `cache_read_per_1m` is the page's "cached input" and
    # `cache_write_per_1m` its "cache writes".
    "gpt-5.6-sol": PriceEntry(
        5.00, 30.00, 0.50, 6.25, as_of=_READ, source=_OPENAI_PRICING),
    "gpt-5.6-terra": PriceEntry(
        2.00, 12.00, 0.20, 2.50, as_of=_READ, source=_OPENAI_PRICING),
}


def price_for(model_id):
    """The `PriceEntry` for `model_id`, or None when the table has no rate.

    None is an answer rather than a fault: the table covers direct models, so a
    routed id comes back None and is costed from the gateway's reported charge
    on the response instead. `is_priced` asks the same question as a bool.
    """
    return PRICES.get(model_id)


def is_priced(model_id):
    """Whether the table carries a rate for `model_id`."""
    return model_id in PRICES


def price_age_days(model_id, today):
    """Days between this entry's reading and `today`, a `datetime.date`.

    The caller passes the date: this module reads no clock, so the age is a
    function of its arguments alone and a recorded one recomputes exactly.
    Raises `KeyError` for a model the table does not price — there is no age to
    report for a reading that was never taken, and zero would look like one
    taken today.
    """
    entry = PRICES.get(model_id)
    if entry is None:
        raise KeyError(
            f"no price entry for {model_id!r}. This table prices the direct "
            f"models; a routed call's cost is read off the response "
            f"(NormalisedResponse.reported_cost). Use `is_priced` or "
            f"`price_for` to ask without raising.")
    return (today - date.fromisoformat(entry.as_of)).days
