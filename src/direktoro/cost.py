"""Cost arithmetic from caller-supplied rates.

The rates come in as an argument, so one function prices every provider and
every rate card: a `direktoro.prices` entry (`price_for(model_id).as_rates()`,
the vendor's published rate with its source and read date attached), or the
caller's own — a negotiated rate, a batch or long-context band, an invoice read
off the bill.

`cost_from_rates` therefore knows nothing about models, providers or the
registry — it multiplies token counts by per-million rates and adds them up.
The one rule it enforces is that UNDER-REPORTING IS IMPOSSIBLE: a counter with
tokens in it and no rate to price
them raises, because the only alternative is to charge zero for work that was
actually billed, and a silent zero is the failure this module exists to
prevent.

That guard has a blind spot, and `cost_from_usage` is what covers it. The rule
above fires on a counter you PASS without a rate; it cannot fire on one you
FORGET to pass, because an omitted counter defaults to zero and zero needs no
rate. Forgetting is the likelier mistake here, because the counter names on a
usage record are not the counter names on a rate card
(`cache_read_input_tokens` / `cache_creation_input_tokens` there,
`cache_read` / `cache_write` here), so every call site that maps one to the
other is a place a counter can quietly go missing. `cost_from_usage` writes
that mapping once, over every counter, so no call site has to.

A cost a provider REPORTS is a different thing and is not computed here: it is
a fact about what was charged, read off the response as
`NormalisedResponse.reported_cost` (gateway-routed calls), and its absence is a
loud fault rather than a zero.
"""

# The counters `cost_from_rates` prices, in the order they appear in the
# returned total. Each is a key the caller's `rates` mapping may carry, in USD
# per MILLION tokens. A caller whose provider bills a counter it never reports
# simply passes no rate for it and no tokens in it.
#
# CACHE WRITES ARE TWO COUNTERS, not one, because they are two prices. Anthropic
# bills a cache write by the entry's time-to-live — a 5-minute write and a
# 1-hour write carry different multipliers over the base input rate — so a
# single "cache_write" rate can only be right for one of them and is silently
# wrong for the other. `"cache_write"` is the 5-minute write (the default TTL);
# `"cache_write_1h"` is the 1-hour one. A provider that has only one kind of
# cache write reports tokens in one counter and needs a rate for only that one.
COUNTERS = ("input", "output", "cache_read", "cache_write", "cache_write_1h")

_PER_MILLION = 1_000_000

# How `cost_from_usage` reads each counter off a usage record: counter name ->
# `NormalisedUsage` attribute. Written once, here, because these two
# vocabularies genuinely differ and a mapping rewritten per call site can
# lose a counter. Enumerating the counters from this mapping rather than naming
# them in the call is the point: a counter cannot be dropped by forgetting to
# mention it. The two cache-creation fields are the per-TTL split; the unsplit
# `cache_creation_input_tokens` total is not priced directly but is checked
# against them (see `cost_from_usage`).
_USAGE_FIELDS = {
    "input": "input_tokens",
    "output": "output_tokens",
    "cache_read": "cache_read_input_tokens",
    "cache_write": "cache_creation_5m_input_tokens",
    "cache_write_1h": "cache_creation_1h_input_tokens",
}


def cost_from_rates(*, rates, input_tokens=0, output_tokens=0,
                    cache_read_tokens=0, cache_write_tokens=0,
                    cache_write_1h_tokens=0):
    """USD cost of one call's usage at the caller's own rates.

    `rates` is a plain mapping of USD per MILLION tokens keyed by counter:
    `"input"`, `"output"`, `"cache_read"`, `"cache_write"`, `"cache_write_1h"`
    (see `COUNTERS`). Anything else in the mapping is ignored, so a caller may
    hand over a wider rate card without filtering it first. Returns an unrounded
    float; callers round for display.

    Token counts follow the normalised usage semantics
    (`direktoro.providers.NormalisedUsage`): `input_tokens` is full-rate,
    cache-miss input ONLY, with cached reads counted separately, so nothing is
    priced twice. `cache_write_tokens` is the 5-minute-TTL cache write and
    `cache_write_1h_tokens` the 1-hour one — two counters because they are two
    prices, and folding them together would price whichever TTL the rate was not
    read for at the wrong multiplier. `cost_from_usage` fills all five from a
    `NormalisedUsage` without the caller naming any of them.

    A NON-ZERO counter with no rate raises `ValueError` naming the counter. It
    is the whole point of taking rates as an argument: those tokens were
    genuinely billed, so pricing them at zero would under-report the total
    silently, and a total that is quietly too low is worse than no total at
    all. A ZERO counter needs no rate — there is nothing to price — so a caller
    passes only the rates its usage actually calls for. Note the shape of what
    this can and cannot catch: it catches a counter handed over unpriced, never
    one that was never handed over at all, since an omitted argument is zero and
    zero asks for no rate. `cost_from_usage` is what closes that side.

    Negative rates and negative token counts raise too. Neither is a thing a
    provider can bill, so both are malformed input, and a negative anywhere in
    this sum reduces the total: exactly the direction that must never happen by
    accident.
    """
    counts = {
        "input": input_tokens,
        "output": output_tokens,
        "cache_read": cache_read_tokens,
        "cache_write": cache_write_tokens,
        "cache_write_1h": cache_write_1h_tokens,
    }

    total = 0.0
    for counter in COUNTERS:
        tokens = counts[counter]
        if tokens < 0:
            raise ValueError(
                f"{counter} token count is negative ({tokens}); token counts "
                f"are counts of work done and cannot be below zero.")
        if not tokens:
            continue
        rate = rates.get(counter)
        if rate is None:
            raise ValueError(
                f"no {counter!r} rate given, but the call reports {tokens} "
                f"{counter} token(s). Those tokens were billed, so costing "
                f"them at zero would under-report the total. Pass "
                f"rates={{{counter!r}: <USD per million tokens>, ...}}, or "
                f"find out why this counter is non-zero before trusting the "
                f"cost.")
        if rate < 0:
            raise ValueError(
                f"{counter!r} rate is negative ({rate}); a rate is USD per "
                f"million tokens and cannot be below zero.")
        total += tokens / _PER_MILLION * rate
    return total


def cost_from_usage(usage, *, rates):
    """USD cost of a `NormalisedUsage` at the caller's own rates.

    The whole of a response's usage, priced, without any call site writing the
    counter mapping itself. `cost_from_rates` is where the arithmetic and the
    missing-rate refusal live; this adds the one thing that function cannot do
    for itself, which is to know that it was handed everything. Its guard fires
    on a counter you pass unpriced and can never fire on a counter you leave
    out, so a hand-written call threading only `input_tokens` and
    `output_tokens` off the record silently prices that call's cached reads and
    cache writes at zero the day a provider starts reporting them. Here the
    counters are enumerated from `_USAGE_FIELDS` rather than named per call, so
    a counter cannot be omitted by forgetting to mention it.

    `usage` is read by attribute, using the `NormalisedUsage` field names, so a
    consumer's own usage record works if it carries the same names — and fails
    with an AttributeError naming the field if it does not, rather than counting
    the absent one as zero.

    CACHE-CREATION TOKENS ARE NEVER LEFT UNATTRIBUTED. Anthropic bills a
    5-minute cache write and a 1-hour cache write at different multipliers
    (1.25x and 2x the base input rate), so `cache_creation_input_tokens` — their
    sum — is not a priceable quantity on its own. This function requires the
    split to account for all of it: if `cache_creation_5m_input_tokens +
    cache_creation_1h_input_tokens` does not equal the unsplit
    `cache_creation_input_tokens`, it raises. It will not price the
    unattributed remainder at zero
    (under-reporting, the failure this module exists to prevent) and it will not
    assign it to a TTL (a guess, priced as though it were a measurement). When a
    provider reports no cache writes at all, all three are zero, the sums agree,
    and there is nothing to attribute.

    That leaves one real case this function deliberately refuses rather than
    handles: a response reporting a cache-write total with no split, where the
    only thing that can resolve the tier is the caller's own knowledge of what
    it asked for. Knowing it, the caller passes the total to `cost_from_rates`
    under the counter it belongs in and gets the right answer; not knowing it,
    there is no right answer to be had here.

    Returns an unrounded float. Raises `ValueError` for a missing rate, a
    negative rate or count, or an unattributed cache-creation remainder.
    """
    counts = {counter: getattr(usage, field)
              for counter, field in _USAGE_FIELDS.items()}
    creation_total = usage.cache_creation_input_tokens
    attributed = counts["cache_write"] + counts["cache_write_1h"]
    if attributed != creation_total:
        raise ValueError(
            f"cache-creation tokens are not fully attributed to a TTL: the "
            f"usage reports {creation_total} cache_creation_input_tokens but "
            f"the per-TTL split accounts for {attributed} "
            f"({counts['cache_write']} at 5m + {counts['cache_write_1h']} at "
            f"1h). A 5-minute write and a 1-hour write are billed at different "
            f"multipliers, so the remainder can be neither costed at zero "
            f"(that under-reports the total) nor assigned to a TTL (that is a "
            f"guess "
            f"priced as a measurement). Fix the usage record so the split sums "
            f"to the total, or find out which TTL the missing tokens were "
            f"written at.")
    return cost_from_rates(
        rates=rates,
        input_tokens=counts["input"],
        output_tokens=counts["output"],
        cache_read_tokens=counts["cache_read"],
        cache_write_tokens=counts["cache_write"],
        cache_write_1h_tokens=counts["cache_write_1h"])
