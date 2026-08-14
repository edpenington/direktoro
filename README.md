# direktoro

One thin adapter per LLM provider, behind a single canonical request and
response shape, plus a model registry that records what each endpoint accepts,
with the evidence and its date beside it. Calling code names a model id and gets
back a normalised response; it never sees a provider's wire format, and it never
discovers a rejected parameter by being billed for the rejection.

**The registry knows how to reach models and what they can do; what they cost is
a separate, dated table.** `direktoro.prices` records each direct model's
published per-million-token rates, and every entry carries the URL of the vendor
page they were read from and the date they were read, so a rate can be traced
and its age measured. Routed models are priced from the gateway's own reported
charge, which comes back on the response. The arithmetic is `cost_from_rates`,
over whatever rates you hand it — a table entry, or your own.

## Requirements

Python 3.11 or newer, and **no third-party runtime dependency beyond the two
provider SDKs** — `anthropic` and `openai`. Nothing else. Everything this
package does apart from making the call is standard library: the registry, the
capability checks, the thinking validation, the cost arithmetic, the routing
serialisation. That is a rule, not an accident, and it is what keeps this layer
from breaking because something underneath it moved.

Both SDKs are imported lazily, inside the adapters, so:

- `import direktoro` succeeds with neither installed;
- the whole registry, capability, thinking and cost surface works with neither
  installed — so a wheel installed `--no-deps` is still useful;
- a run that only calls Anthropic models never needs `openai` present.

CI checks that against a built wheel in an environment with both SDKs genuinely
absent.

## Install

```sh
pip install direktoro
```

Or from a checkout: `pip install -e ".[dev]"`.

## Usage

Ask the registry what a model is and what it takes, and get back exactly what
will go on the wire:

```python
from direktoro import (
    Thinking, model_info, resolved_decoding_params, supports_forced_tool_choice,
    thinking_support,
)

info = model_info("claude-opus-5")
info.provider, info.api_key_env
# ('anthropic', 'ANTHROPIC_API_KEY')

supports_forced_tool_choice("claude-opus-5")   # True
thinking_support("claude-opus-5").default_on   # True — it thinks unless told not to

resolved_decoding_params(
    "claude-opus-5", sampling={"temperature": 0.0}, max_tokens=8192,
    thinking=Thinking(mode="adaptive", effort="high"))
# {'max_tokens': 8192, 'output_config': {'effort': 'high'}, 'thinking': {'type': 'adaptive'}}
```

Note what is missing from that last result: `temperature`. Opus 4.7 and later
reject it with a 400, so the resolver omits it — from the wire request *and*
from the record of the call, so a config temperature never moves the identity of
a model that would have ignored it.

A shape the endpoint would reject is refused before there is a client to send it
with:

```python
from direktoro import ThinkingUnsupported

resolved_decoding_params(
    "claude-opus-5", sampling={"temperature": 0.0}, max_tokens=8192,
    thinking=Thinking(mode="disabled", effort="max"))
# ThinkingUnsupported: model 'claude-opus-5' accepts `thinking={'type': 'disabled'}`
# only at effort 'high' or below, and this call is at 'max'. The pair returns a
# 400. Lower the effort, or leave thinking enabled.
```

Making the call. The model id picks the provider, the base URL, the key
environment variable and the wire format, so there is nothing else to configure:

```python
from direktoro import build_adapter, extract_tool_call, tool_choice_named

RECORD = {
    "name": "record_answer",
    "description": "Record your answer. Call this tool exactly once.",
    "input_schema": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    },
}

adapter = build_adapter("claude-opus-5")          # reads ANTHROPIC_API_KEY
response = adapter.create_message(
    model="claude-opus-5",
    system=[{"type": "text", "text": "Answer by calling the tool."}],
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    tools=[RECORD],
    tool_choice=tool_choice_named("claude-opus-5", "record_answer"),
    max_tokens=8192,
)

answer, error = extract_tool_call(response, "record_answer")
# ({'answer': 'Paris'}, None)
response.usage.input_tokens, response.usage.output_tokens
```

Swap the model id for `gpt-5.6-sol` or `z-ai/glm-4.6v` and none of the rest
changes: `tool_choice_named` reshapes the forced choice per wire protocol (or
degrades it to `auto` for an endpoint that cannot be forced), the adapter
translates the request out and the response back, and `response.content` is the
same list of blocks either way.

Costing it, at rates you supply:

```python
from direktoro import cost_from_rates

cost_from_rates(
    rates={"input": 5.00, "output": 25.00, "cache_read": 0.50},  # USD per Mtok
    input_tokens=12_000, output_tokens=800, cache_read_tokens=40_000)
# 0.1

cost_from_rates(rates={"input": 5.00}, output_tokens=800)
# ValueError: no 'output' rate given, but the call reports 800 output token(s).
# Those tokens were billed, so costing them at zero would under-report the total.
```

That refusal is the point of the whole arrangement: a total that is quietly too
low is the one nobody questions.

It can only fire on a counter you *pass*, though. A counter you forget to pass
defaults to zero, and zero needs no rate — so hand-mapping a usage record onto
those arguments is where a cached read starts riding free. `cost_from_usage`
does the mapping once, over every counter:

```python
from direktoro import cost_from_usage

# `response` is the one returned by the call above.
cost_from_usage(response.usage,
                rates={"input": 5.00, "output": 25.00, "cache_read": 0.50})
```

Cache writes are two counters, not one: Anthropic bills a 5-minute write at
1.25x the base input rate and a 1-hour write at 2x, so `rates` takes
`cache_write` and `cache_write_1h` separately. If a usage record's per-TTL split
does not account for its cache-creation total, `cost_from_usage` raises rather
than costing the remainder at zero or guessing which tier it belonged to.

The vendors' published rates, dated, are in `direktoro.prices`:

```python
from datetime import date

from direktoro import cost_from_usage, price_age_days, price_for

entry = price_for("claude-opus-5")
entry.as_rates()
# {'input': 5.0, 'output': 25.0, 'cache_read': 0.5, 'cache_write': 6.25}
entry.as_of, entry.source
# ('2026-08-07', 'https://platform.claude.com/docs/en/about-claude/pricing')

price_age_days("claude-opus-5", date(2026, 8, 21))   # 14 days since it was read

cost_from_usage(response.usage, rates=entry.as_rates())
```

`price_for` returns `None` for a routed model, whose cost comes back from the
gateway, and `PRICES_VERSION` identifies the table's data, so a run can record
which table priced it.

## What is in it

- **Provider adapters** — `AnthropicAdapter` and `OpenAIAdapter`. Callers
  speak this package's canonical format (system as a string or text blocks,
  messages as content-block lists, tools with an `input_schema` — a block
  vocabulary that deliberately coincides with the Anthropic wire); adapters
  translate outward. The OpenAI adapter covers two wire protocols, the
  Responses API and the OpenAI-compatible Chat Completions surface, chosen
  per model by the registry. `create_message` makes exactly one call and
  raises a normalised `ProviderError` (or a retryable subclass) on failure;
  `create_message_with_retry` is the backoff loop for callers who want one.

- **Audit-log helpers** — `direktoro.wire_log`: `redact_wire_request` (and
  `redact_messages` / `redact_system`) stub inline base64 image bytes into
  `image_ref` records across every wire image shape this package speaks, and
  `response_to_dict` flattens a provider SDK response object. Every adapter
  response carries the `wire_request` actually sent and a plain-dict
  `raw_response`, so a consumer's audit log is redact-and-write.

- **Model registry** — 16 entries: eleven direct (Anthropic and OpenAI) and
  five routed through OpenRouter. Each records provider, base URL, key
  environment variable, wire protocol, and capability flags for vision, forced
  tool choice and sampling controls, including the documented value range per
  sampling param where one is published. **Vision and forced tool choice are
  stated per entry and have no default at all**, because a default there is a
  capability claim: an entry that says nothing about image input would
  otherwise report as multi-modal, and a consumer reading it would send image
  parts and be billed for the rejection. Every entry is vision-capable today,
  which is a fact about the table rather than something a new entry inherits.
  Twelve of the sixteen declare a thinking surface: the six live Anthropic
  entries and both GPT-5.6 entries from the published reference, and four
  routed entries from live probes — one of which declares an *empty* surface,
  an instruct endpoint probed to take no reasoning parameter at all. On the
  other four the surface is *undeclared*, which means direktoro refuses to emit
  a thinking shape for them rather than guessing one, not that they cannot
  think: three are retired ids nobody can re-verify, and one is a routed entry
  whose probe was rate-limited and whose gateway page does not enumerate the
  levels. An unknown id raises and lists what is known, separating the ids that
  can start a new run from the retired ones that resolve only so past runs stay
  citable.

- **Price table** — `direktoro.prices`, a dated reading of the vendors'
  published rates for the direct models: input, output, cached reads and
  5-minute cache writes, in USD per million tokens, each entry carrying the page
  it was read from and the date. `price_for(model_id).as_rates()` is exactly
  what `cost_from_rates` takes, `price_age_days(model_id, today)` measures the
  reading's age against a date you pass, and `PRICES_VERSION` names the table's
  data so a run can record which one priced it. Long-context bands and the
  1-hour cache-write tier are yours to supply; the table is the standard tier
  and the 5-minute write.

- **Retirement** — a withdrawn id keeps its entry, because the registry is also
  the provenance record for runs that already happened, and deleting it would
  leave a stored model id unresolvable. Nothing here refuses such an id on your
  behalf: every lookup and every adapter keeps resolving one. The gate is yours
  to apply, wherever you accept a new run — `is_retired(model_id)`, or
  `known_models(include_retired=False)` for the startable list.
  `direktoro-smoke` applies them at config load.

- **Thinking and reasoning effort** — a per-call `Thinking` spec validated
  against the registry's per-model `ThinkingSupport` and rendered for the
  model's wire: Anthropic's `thinking` / `output_config` keys, or the single
  OpenAI-family reasoning level, where disabling rides as `"none"`. Because the
  registry knows each model's surface, a request the endpoint would answer with
  a 400 — `budget_tokens` on a family that removed it, an effort level a model
  does not have, a temperature on a request that also turns thinking on —
  raises `ThinkingUnsupported` before a client exists; so does a `max_tokens` a
  thinking call cannot answer within, which the endpoint would accept and then
  spend entirely on reasoning. Effort and mode reach the call-identity block
  through the decoding params, so two runs differing only in effort record
  differently.

- **Routing and provenance** — a routed entry carries a `Route` pinning the
  upstream provider, refusing fallbacks, refusing endpoints that would silently
  drop a parameter, and requesting zero data retention. The adapter emits it,
  captures the gateway's generation id and reported cost, and asserts the pin
  against the provider that actually served the response. A call that did not go
  where it declared, or that came back without its receipt, raises rather than
  being recorded.

- **Call identity** — `call_identity_fields` returns the canonical, byte-stable
  block describing the provider side of a call: model, provider, base URL,
  route, and the decoding params keyed under the *wire's* parameter name.
  Compose it with your own prompt and template hashes; the composed fingerprint
  stays yours. `source_hash()` digests this package's own `.py` files into one
  sha256, for a record that names the engine's bytes as well as its version.

- **Batch** — a minimal Anthropic Message Batches client for long runs of
  independent requests, normalising each result into the same shape a live call
  returns, and failing loudly on any request that did not come back.

- **`direktoro-smoke`** — a cross-provider plumbing check: one forced tool call
  per provider. It dry-runs by default, printing each canonical request without
  a client, a key or a network call. `--live` spends real money.

- **`py.typed`** — the package ships the PEP 561 marker, so a type checker
  reads the annotations on the dataclasses a consumer actually holds — `Model`,
  `Thinking`, `ThinkingSupport`, `NormalisedResponse`, `NormalisedUsage`,
  `Route` — rather than treating every import as untyped.

## The identity block is a compatibility surface

`call_identity_fields`, the `fingerprint_fields` block nested inside it, the
`resolved_decoding_params` mapping it embeds, and the bytes `canonical_json`
produces from any of them are what a consumer hashes into the provenance it
publishes. A consumer that has published a number cannot recompute it: the run
is over, and the fingerprint recorded beside it is now the only claim that two
runs were the same run.

So a change to the shape of those outputs — a key added, removed or renamed, a
list reordered, a value spelled differently or omitted under different
conditions — is a breaking change, in the same class as deleting a public
function: it moves every number every consumer has already published, and it
takes a major version. That holds even when the change is otherwise invisible:
a key that appears only for models nobody uses still re-fingerprints the models
everybody does, if it changes the serialised form. There is no such thing as a
cosmetic edit to this block.

Adding a new model entry, or changing what an existing entry's identity block
says because the provider changed — a re-pinned upstream, a withdrawn
parameter — is not a break: the block tracks what is actually sent, and a run
under a new configuration should fingerprint differently. What must not move
is the shape into which a given configuration is rendered.

## How the registry stays honest

Every capability flag and quirk carries its evidence, in a comment beside it,
with the date.

**The standard is a clear published statement, not a probe.** If the vendor's or
the gateway's own reference says a model takes an input, a parameter or a value,
that is sufficient to record it: write the value, cite the page and the date it
was read, and move on. Calling an endpoint to watch it accept each parameter is
not the bar — it costs money and time, and its result goes stale as fast as the
documentation does. An entry left half-finished because the probing was
expensive blocks a model that works, which is the more common failure by far.
When a documented value does turn out to be wrong, the live call is the
correction: raise it, fix that entry, record what the call did. It is not a
reason to re-probe everything else.

What a probe is *for* is the case documentation does not settle — a
gateway-served endpoint whose upstream diverges from the model's own reference,
a value the reference declines to enumerate, a behaviour nobody wrote down. So
what the table rests on today is:

- **The eleven direct entries** rest on the vendor's **published model
  reference** — the model and deprecation tables, the migration guide, the
  thinking and reasoning documentation, the API reference. No Anthropic or
  OpenAI endpoint was called to establish any of them, and none needs to be.
- **The five routed entries** rest mostly on **live endpoint probes** — the
  gateway's `/models` and `/endpoints` listings for served upstream,
  quantization and supported parameters, then a real plain / tool / vision call
  against the pinned endpoint. A pinned upstream's behaviour is exactly what the
  gateway's model page does not tell you, and two hosts serving one slug can
  disagree. Where the gateway's own documented request surface settles the
  question instead (the sampling bands, the effort mapping), the comment says so
  and no probe was run.

Every value's comment carries the date its evidence was read or probed; the
dates currently in the table run from 2026-07-23 to 2026-08-14.

A value left at its field default records nothing at all — so a field whose
default would read as a *claim* does not get one. Two are in that position and
neither has a working default: `supports_images` and `forced_tool_choice`. Every
entry states both, with its basis beside it, and an unstated one is a
construction error rather than a silent claim. The remaining defaults are honest
absences: an empty `rejects_sampling` claims no refusal, an empty
`sampling_bands` claims no range, a `None` thinking surface claims nothing at
all.

What is still forbidden is copying a flag across because the entry above it sets
the same one. Families are not uniform — two upstreams serving one slug can
disagree about whether a forced tool choice routes, and a vendor can drop the
sampling controls in a point release — so a flag that differs from its
neighbours is not an inconsistency to tidy away: changing it means re-probing
the endpoint or re-reading the reference. A shared family constant is not that:
the published reference states those facts per family, and the constant carries
the citation and its read date.

The standing limitation is that this is hand-verified and nothing re-checks it
on a schedule. A model whose behaviour changes without an announcement will be
wrong here until someone looks. Entries carry their dates so the gap is visible
rather than buried.

## Development

```sh
pip install -e ".[dev]"
pytest -q
```

The suite is hermetic. A session-wide network guard (`tests/conftest.py`) fails
any test that opens a non-local connection or resolves a non-local name, so no
test needs an API key and none can spend money. The source distribution ships
the suite complete, guard included, and CI runs the published copy in a clean
directory containing nothing but pytest — a shipped suite that reached the
network would contradict the property it exists to demonstrate.

## License

Apache-2.0 — see `LICENSE`.

direktoro is a support library and carries no citation of its own: cite the
project that used it. meltiro records the direktoro version in every run record
it publishes.
