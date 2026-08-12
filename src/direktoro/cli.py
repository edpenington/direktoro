"""Cross-provider live-smoke harness: one forced tool call per provider.

This is a plumbing check for the shared provider layer, not an accuracy
evaluation. For one representative model per real provider (or an explicit
`--models` list) it proves, per model, that the API key resolves, the adapter
builds the right wire request, the model returns a forced tool call, and usage
normalises. It answers "does the wiring work end to end", nothing about whether
any answer is good.

The provider-generic machinery lives here, so a consumer that wants to smoke
its own prompts keeps only its prompt rendering.

Usage:

    direktoro-smoke                 # dry run: print each canonical request
    direktoro-smoke --live          # spends real money on every provider

`--dry-run` is the default: it renders and prints each model's canonical
request and exits without any API call, which proves the request-building for
every provider for free. A live run needs the explicit `--live` flag AND each
selected provider's API key exported into the environment. python-dotenv is
deliberately not a dependency, so load a .env yourself first:

    set -a; source .env; set +a

Exit status: non-zero when any selected model did not return a clean tool call
(missing key, API error after retries, no tool call, empty answer, or a routed
response that came back without the gateway's own cost figure). A dry run exits
zero unless `--thinking` / `--effort` named a shape some selected model's
endpoint would reject, which the registry refuses up front and reports per
model.

Cost. Only a gateway-routed call reports what it was charged, and that figure
is recorded verbatim in the table. A direct call reports no cost and this
harness computes none: the column holds what was charged, and a figure priced
from a rate card is the caller's arithmetic (`direktoro.prices` +
`direktoro.cost.cost_from_rates`) over the rates it chooses.
"""

import argparse
import dataclasses
import json
import sys
import time

from direktoro.providers import (
    MissingAPIKey, Thinking, ThinkingUnsupported, build_adapter,
    create_message_with_retry, extract_tool_call, resolved_decoding_params,
    tool_choice_named)
from direktoro.registry import (
    EFFORT_LEVELS, MODEL_REGISTRY, PROVIDER_ANTHROPIC, SAMPLING_PARAMS,
    THINKING_MODES, WIRE_RESPONSES, is_retired, model_info)


# The single forced tool. A plumbing placeholder: one required string field is
# enough to prove the tool call round-trips through every wire dialect.
TOOL_NAME = "record_answer"
RECORD_ANSWER_TOOL = {
    "name": TOOL_NAME,
    "description": (
        "Record your answer to the question. Call this tool exactly once."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer, as a short string.",
            },
        },
        "required": ["answer"],
    },
}

SYSTEM_TEXT = (
    "You are part of an automated plumbing check for an LLM provider "
    "abstraction layer. Answer the user's question by calling the "
    "record_answer tool exactly once."
)
USER_TEXT = "What is the capital of France?"

# Output-token cap per call. Sized for a model that thinks by default: on Claude
# Opus 5 and Sonnet 5 thinking runs when the `thinking` parameter is omitted, and
# max_tokens caps thinking PLUS response text, so a cap tuned on a non-thinking
# model can be spent entirely on thinking and truncate the tool call this harness
# is checking for. This is a CAP, not a spend — raising it cannot generate tokens
# a model would not otherwise have generated — so the bill for a model that does
# not think is unaffected.
DEFAULT_MAX_TOKENS = 8192

# No default temperature. `--temperature` is omitted unless asked for, exactly
# like `--thinking`, which leaves each model's own sampling default in force.
# A default of 0.0 would be a decision this harness has no reason to make and
# one consequence it cannot escape: on the 4.6-generation and earlier endpoints
# a temperature and active thinking cannot both be sent (the pair is a 400, so
# `resolved_decoding_params` refuses it), and those are precisely the entries a
# `--thinking` run would exercise. With a temperature always present, every such
# run is refused before it starts and `--thinking-budget` — whose only eligible
# model is the pre-4.6 Haiku 4.5 — can never be sent at all. Pass
# `--temperature 0.0` to pin it deliberately; a plumbing check that only counts
# tool calls does not need it pinned.
DEFAULT_TEMPERATURE = None

_KEY_SUFFIX = "_API_KEY"


# ---------------------------------------------------------------------------
# Model selection (programmatic, from the registry)
# ---------------------------------------------------------------------------

def provider_label_for_env(env):
    """Human provider label derived from an api_key_env, e.g. ANTHROPIC_API_KEY
    becomes 'anthropic'. Derived, so a registry rename can never leave a stale
    hardcoded label behind."""
    base = env[:-len(_KEY_SUFFIX)] if env.endswith(_KEY_SUFFIX) else env
    return base.lower()


def provider_label(model_id):
    """Provider label for a model id (via its api_key_env)."""
    return provider_label_for_env(model_info(model_id).api_key_env)


# The model `--live` reaches for when the caller names none, per key env.
# NAMED rather than derived: a derived choice has to rank the entries by
# something, and nothing available here ranks them. Sorting by id picks
# whichever slug happens to sort first, and the price table covers the direct
# entries only, so a derived pick could silently move a live, billable run onto
# a different model as entries are added. Each of these is the smallest tool-calling model its
# provider offers, which is what a reachability probe wants: the cheapest
# possible round trip that still exercises the adapter, the wire translation
# and a tool call.
_SMOKE_REPRESENTATIVES = {
    "ANTHROPIC_API_KEY": "claude-haiku-4-5-20251001",
    "OPENAI_API_KEY": "gpt-5.6-terra",
    "OPENROUTER_API_KEY": "google/gemini-3.6-flash",
}


def select_default_models():
    """One representative model per real provider.

    Real providers are keyed by api_key_env rather than by the `provider`
    field: every gateway-served model shares one provider string and one key,
    so the key env is what distinguishes the parties actually being billed. The
    representative for each is named in `_SMOKE_REPRESENTATIVES`, so the model a
    `--live` run bills is
    a reviewed choice rather than a consequence of how ids sort. A key env with
    no named representative, or one naming a model that is absent or retired,
    fails here: adding a provider forces the choice to be made rather than
    defaulting to whatever the registry happens to yield first. A caller who
    wants something else names it with `--models`.

    Every registered model supports tool use (the registry holds only
    tool-calling models), so the "supports tool use" filter is registry
    membership itself.

    Returns a list of (provider_label, model_id) in the registry's provider
    order. This text-only harness sends no images, so an image-incapable entry
    is fine here; a provider whose key is unset at run time simply records a
    MissingAPIKey cell rather than aborting the matrix.
    """
    order = []  # env order by first appearance in the registry
    for _model_id, info in MODEL_REGISTRY.items():
        if not info.retired and info.api_key_env not in order:
            order.append(info.api_key_env)

    selected = []
    for env in order:
        model_id = _SMOKE_REPRESENTATIVES.get(env)
        if model_id is None:
            raise ValueError(
                f"no smoke representative named for {env}. Add one to "
                f"_SMOKE_REPRESENTATIVES: the model a --live run reaches for "
                f"is a billable choice and is not derived.")
        info = MODEL_REGISTRY.get(model_id)
        if info is None or info.retired:
            raise ValueError(
                f"the smoke representative for {env} is {model_id!r}, which is "
                f"{'retired' if info else 'not in the registry'}. Name a live "
                f"model in _SMOKE_REPRESENTATIVES.")
        selected.append((provider_label_for_env(env), model_id))
    return selected


def resolve_models(models_arg):
    """Resolve the --models override, or fall back to the default per-provider
    selection. Every id is validated against the registry: unknown ids fail
    loudly, and so do RETIRED ids.

    This is where THIS program applies the new-run gate that
    `registry.Model.retired` exists for — `registry.is_retired`, at config load,
    so the failure is loud and up front rather than a 404 part-way through a
    run. The gate lives in the caller because it has to: `model_info` and the
    adapters keep resolving a retired id, or a run that already happened would
    stop being resolvable and citable. A library consumer applies the same
    predicate at its own equivalent boundary.

    Returns a list of (provider_label, model_id)."""
    if not models_arg:
        return select_default_models()
    resolved = []
    for raw in models_arg.split(","):
        model_id = raw.strip()
        if not model_id:
            continue
        model_info(model_id)  # raises ValueError for an unknown id
        if is_retired(model_id):
            raise ValueError(
                f"model '{model_id}' is retired: the provider has withdrawn "
                f"it, so a live call would fail. Its registry entry is kept "
                f"only so past runs still resolve against it. Name a current "
                f"model instead.")
        resolved.append((provider_label(model_id), model_id))
    if not resolved:
        raise ValueError("--models was given but named no models")
    return resolved


# ---------------------------------------------------------------------------
# Canonical request
# ---------------------------------------------------------------------------

def wire_protocol_label(model_id):
    """The wire protocol the adapter speaks for a model."""
    info = model_info(model_id)
    if info.provider == PROVIDER_ANTHROPIC:
        return "anthropic-messages"
    if info.wire_api == WIRE_RESPONSES:
        return "openai-responses"
    return "openai-chat-completions"


def thinking_from_args(args):
    """The `Thinking` spec the flags ask for, or None when neither was given.

    None sends no thinking parameters at all, leaving each model's own default
    in force — which is what a plumbing check should exercise by default."""
    mode = getattr(args, "thinking", None)
    effort = getattr(args, "effort", None)
    budget = getattr(args, "thinking_budget", None)
    if mode is None and effort is None:
        return None
    return Thinking(mode=mode, effort=effort, budget_tokens=budget)


def _sampling_from_args(args):
    """The sampling controls this invocation specified, as a mapping.

    One flag today (`--temperature`), and absent unless given: the smoke run's
    job is to exercise what a caller asked for, so an unasked-for control is
    left to the model's own default rather than pinned to a value nobody
    chose. A second flag would add a key here and nothing else.
    """
    return {"temperature": args.temperature}


def build_request(model_id, *, max_tokens, sampling=None, thinking=None):
    """The canonical request kwargs for one model, ready to splat into
    adapter.create_message. Anthropic-shaped throughout except tool_choice,
    which `tool_choice_named` shapes per wire protocol.

    The sampling controls ride as ONE mapping under `sampling`, because that is
    the parameter every adapter takes (`create_message(..., sampling={...})`);
    each control's own name is a keyword no adapter defines, so a request that
    spread them flat would be a TypeError on every model the moment it was
    splatted.

    An unspecified sampling control and an unspecified `thinking` are each
    omitted ENTIRELY rather than sent as a null, and a sampling mapping left
    with nothing in it is omitted too: an absent key is what "leave the model's
    own default in force" looks like on the wire, and it is also what the
    printed request should show, since a `"temperature": null` in a dry run
    reads as a parameter being sent when none is."""
    request = {
        "model": model_id,
        "system": [{"type": "text", "text": SYSTEM_TEXT}],
        "messages": [{"role": "user", "content": USER_TEXT}],
        "tools": [RECORD_ANSWER_TOOL],
        "tool_choice": tool_choice_named(model_id, TOOL_NAME),
        "max_tokens": max_tokens,
    }
    specified = {name: value for name, value in (sampling or {}).items()
                 if value is not None}
    if specified:
        request["sampling"] = specified
    if thinking is not None:
        request["thinking"] = thinking
    return request


def _sampling_as_resolved(resolved):
    """The sampling controls the resolver kept, read back off the resolved
    decoding params.

    The dry run builds its printed request from this rather than from the raw
    ask, so the request and the resolved params printed above it agree: a
    control the model refuses is missing from both. The printed request is
    the CANONICAL request — `create_message` kwargs, sampling nested,
    thinking as the caller's spec — not the wire; the `resolved decoding
    params` line above it is the wire-keyed record of what the decoding side
    of the wire carries.
    """
    return {name: resolved[name] for name in SAMPLING_PARAMS
            if name in resolved}


def _jsonable(request):
    """The canonical request with its `Thinking` spec rendered as a plain dict,
    so the dry run can print it. Only the display path needs this; the live path
    passes the dataclass straight to the adapter."""
    thinking = request.get("thinking")
    if thinking is None:
        return request
    shown = dict(request)
    shown["thinking"] = {k: v for k, v in dataclasses.asdict(thinking).items()
                         if v is not None}
    return shown


def check_answer(tool_input):
    """Return a list of violations for a record_answer input. Empty means
    clean: the input is an object whose `answer` is a non-empty string."""
    if not isinstance(tool_input, dict):
        return [f"tool input is not an object: {tool_input!r}"]
    answer = tool_input.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return ["answer is missing or empty"]
    return []


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def run_dry(models, args):
    """Print each model's provider, wire protocol, resolved decoding params,
    and full canonical request as JSON. No client, no key, no network.

    The `resolved decoding params` line is the wire-keyed record of what
    `--thinking` / `--effort` / a sampling flag put on the decoding side of
    the wire (registry defaults included); the request printed under it is
    the CANONICAL request the live path would hand to `create_message`. The
    dry run is also the free way to see the resolver REFUSE a shape a model
    would reject — a thinking spec its surface does not take, a sampling
    value outside its documented band, a cap a reasoning call cannot answer
    within. Each refusal is a per-model fact, so it is reported per model and
    the run continues with a non-zero exit, rather than aborting the whole
    matrix."""
    thinking = thinking_from_args(args)
    refused = 0
    for label, model_id in models:
        info = model_info(model_id)
        print(f"=== {label}: {model_id}")
        print(f"    provider: {info.provider}   "
              f"wire: {wire_protocol_label(model_id)}   "
              f"base_url: {info.base_url}")
        try:
            resolved = resolved_decoding_params(
                model_id, sampling=_sampling_from_args(args),
                max_tokens=args.max_tokens, thinking=thinking)
        except ValueError as e:
            # ThinkingUnsupported is a ValueError; the band and cap refusals
            # are plain ones. All are per-model config infeasibility.
            refused += 1
            print(f"    REFUSED: {e}")
            print()
            continue
        request = build_request(
            model_id, max_tokens=args.max_tokens,
            sampling=_sampling_as_resolved(resolved), thinking=thinking)
        print(f"    resolved decoding params: {json.dumps(resolved)}")
        print(json.dumps(_jsonable(request), indent=2, ensure_ascii=False))
        print()
    return 1 if refused else 0


# ---------------------------------------------------------------------------
# Live run
# ---------------------------------------------------------------------------

def call_one(label, model_id, args):
    """Run one model live and return a result row. Never raises: a provider
    failure — including a routed response that arrived without the gateway's
    cost figure, exactly the kind of plumbing fault this harness exists to
    surface — is captured in the row so the matrix continues."""
    row = {
        "label": label, "model": model_id, "ok": False, "answer": None,
        "input_tokens": None, "output_tokens": None, "cache_read": None,
        "cache_write": None, "cost": None, "wall": None, "stop_reason": None,
        "error": None, "violations": None,
        "generation_id": None, "served_provider": None,
    }
    try:
        adapter = build_adapter(model_id)
    except MissingAPIKey as e:
        row["error"] = f"{type(e).__name__}: {e}"
        return row

    request = build_request(
        model_id, max_tokens=args.max_tokens,
        sampling=_sampling_from_args(args),
        thinking=thinking_from_args(args))

    t0 = time.monotonic()
    try:
        response = create_message_with_retry(adapter, **request)
    except Exception as e:
        # ThinkingUnsupported lands here too, raised by the resolver before any
        # request is sent: the row records the refusal and no money is spent.
        row["wall"] = time.monotonic() - t0
        row["error"] = f"{type(e).__name__}: {e}"
        return row
    row["wall"] = time.monotonic() - t0

    row["stop_reason"] = response.stop_reason
    usage = response.usage
    row["input_tokens"] = usage.input_tokens
    row["output_tokens"] = usage.output_tokens
    row["cache_read"] = usage.cache_read_input_tokens
    row["cache_write"] = usage.cache_creation_input_tokens
    row["generation_id"] = response.generation_id
    row["served_provider"] = response.served_provider
    if model_info(model_id).route is not None:
        # Routed model: the gateway reports what the call was charged
        # (OpenRouter usage.cost) and the row records that figure as the fact it
        # is. A routed response with no reported cost is a plumbing fault worth
        # surfacing. A direct call reports no cost and the row leaves it None:
        # this column is what was CHARGED, and a figure computed from a rate
        # card is a different fact (`direktoro.prices` + `cost_from_rates` is
        # where a caller makes one).
        row["cost"] = response.reported_cost
        if row["cost"] is None:
            row["error"] = ("routed response carried no reported cost "
                            "(usage.cost); did usage.include reach the gateway?")
            return row

    tool_input, err = extract_tool_call(response, TOOL_NAME)
    if err:
        row["violations"] = [err]
        return row
    violations = check_answer(tool_input)
    if violations:
        row["violations"] = violations
        return row
    row["answer"] = tool_input["answer"]
    row["ok"] = True
    return row


def _fmt_int(value):
    return "" if value is None else str(value)


def print_live_row(row):
    """Print one model's live result: status, answer or loud violation report,
    stop reason, tokens, and the reported cost where the provider gave one."""
    if row["error"]:
        print(f"    status: FAIL ({row['error']})")
        return
    status = "OK" if row["ok"] else "FAIL"
    print(f"    status: {status}   stop_reason: {row['stop_reason']!r}")
    print(f"    tokens: in={row['input_tokens']} out={row['output_tokens']} "
          f"cache_read={row['cache_read']} cache_write={row['cache_write']}")
    if row.get("served_provider") is not None:
        print(f"    routed: served_provider={row['served_provider']!r} "
              f"generation_id={row['generation_id']!r}")
    if row["cost"] is not None:
        print(f"    reported cost USD: {row['cost']:.6f}")
    if row["violations"]:
        print("    TOOL-CALL VIOLATION:")
        for v in row["violations"]:
            print(f"      - {v}")
    if row["answer"] is not None:
        print(f"    answer: {row['answer']!r}")


def print_table(rows):
    """Print the final cross-provider table.

    The cost column and its total cover only what the providers themselves
    reported, so a run of direct models totals zero: that is an absence of
    reported figures, not a claim that the calls were free."""
    header = (f"{'model':<28} {'status':<7} {'in':>7} "
              f"{'out':>7} {'reported $':>11} {'secs':>6}")
    print(header)
    print("-" * len(header))
    total = 0.0
    for row in rows:
        total += row["cost"] or 0.0
        status = "ok" if row["ok"] else "FAIL"
        cost = "" if row["cost"] is None else f"{row['cost']:.6f}"
        secs = "" if row["wall"] is None else f"{row['wall']:.1f}"
        print(f"{row['model']:<28} {status:<7} "
              f"{_fmt_int(row['input_tokens']):>7} "
              f"{_fmt_int(row['output_tokens']):>7} {cost:>11} {secs:>6}")
    print("-" * len(header))
    print(f"{'TOTAL':<28} {'':<7} {'':>7} {'':>7} {total:>11.6f}")


def run_live(models, args):
    rows = []
    for label, model_id in models:
        print(f"=== {label}: {model_id}", flush=True)
        row = call_one(label, model_id, args)
        print_live_row(row)
        print(flush=True)
        rows.append(row)
    print_table(rows)
    return 1 if any(not row["ok"] for row in rows) else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _positive_int(text):
    """argparse type for --max-tokens: a cap below one is malformed input."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(
            f"must be a positive integer, got {value}")
    return value


def build_arg_parser():
    p = argparse.ArgumentParser(
        prog="direktoro-smoke",
        description=__doc__.splitlines()[0],
        epilog=(
            "A live run spends real money on all selected providers and needs "
            "each provider's API key exported into the environment. "
            "python-dotenv is not a dependency, so load a .env yourself "
            "first:\n\n    set -a; source .env; set +a\n"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--models", default=None, metavar="id,id,...",
        help="Comma-separated model ids to run, overriding the default one per "
             "provider. Every id must be in the registry.")
    p.add_argument(
        "--max-tokens", type=_positive_int, default=DEFAULT_MAX_TOKENS,
        help=f"Output-token cap per call (default {DEFAULT_MAX_TOKENS}).")
    p.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help="Decoding temperature. Omitted by default, leaving each model's "
             "own sampling default in force; dropped from the wire request AND "
             "from the recorded call identity for a model whose registry entry "
             "names it in `rejects_sampling`, and refused before any spend when "
             "it falls outside the band that entry documents. Sending one rules "
             "out --thinking on the models that accept both parameters "
             "individually but reject the pair (Sonnet 4.6, Haiku 4.5), so "
             "leave it off to exercise thinking.")
    p.add_argument(
        "--thinking", choices=THINKING_MODES, default=None,
        help="Thinking mode to request. Omitted by default, which leaves each "
             "model's own default in force (Claude Opus 5 and Sonnet 5 think "
             "by default; Opus 4.8/4.7 do not). A shape the model's endpoint "
             "would reject is refused before the call, not after.")
    p.add_argument(
        "--effort", choices=EFFORT_LEVELS, default=None,
        help="Reasoning effort to request, rendered for the model's wire "
             "(Anthropic output_config.effort; the single OpenAI-family "
             "reasoning level elsewhere). Omitted by default. Levels a "
             "model's entry does not declare are refused.")
    p.add_argument(
        "--thinking-budget", type=_positive_int, default=None, metavar="N",
        help="Thinking-token budget for --thinking budget (pre-4.6 models "
             "only; must be >= 1024 and < --max-tokens).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--live", action="store_true",
        help="Make real API calls. Spends real money.")
    mode.add_argument(
        "--dry-run", action="store_true",
        help="Print the canonical request per model and exit (the default).")
    return p


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        models = resolve_models(args.models)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Vocabulary errors in the thinking flags (a budget without budget mode, a
    # display without adaptive) are the caller's mistake, not a model
    # capability: report them here, before any model is touched.
    try:
        thinking_from_args(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.live:
        return run_live(models, args)
    return run_dry(models, args)


if __name__ == "__main__":
    sys.exit(main())
