"""direktoro: a shared LLM-provider layer.

One provider abstraction, one model registry, one routing and provenance
discipline. The names re-exported here are the stable public surface a
downstream consumer imports; import them from the package root, e.g.
`from direktoro import build_adapter`.

The registry knows how to REACH a model and what that model can do. Unit prices
are a separate seam: `direktoro.prices` is a dated table recording each vendor's
published rate with the page it was read from and the date it was read,
`cost_from_rates` does the arithmetic over rates the caller supplies (a table
entry, or its own), and a gateway-reported charge is recorded off the response
(`NormalisedResponse.reported_cost`) as the fact it is.

The `anthropic` and `openai` SDKs are imported lazily inside the adapters, so
`import direktoro` succeeds without either SDK installed.
"""

from direktoro.providers import (
    AnthropicAdapter,
    MissingAPIKey,
    NormalisedResponse,
    NormalisedUsage,
    OpenAIAdapter,
    ProviderError,
    ProviderRateLimitError,
    ProviderRetryableError,
    RETRY_BACKOFF_SECONDS,
    Thinking,
    ThinkingUnsupported,
    build_adapter,
    create_message_with_retry,
    extract_tool_call,
    resolved_decoding_params,
    split_decoding_config,
    tool_choice_named,
)
from direktoro.cost import cost_from_rates, cost_from_usage
from direktoro.prices import (
    PRICES,
    PRICES_VERSION,
    PriceEntry,
    is_priced,
    price_age_days,
    price_for,
)
from direktoro.provenance import source_hash
from direktoro.registry import (
    EFFORT_LEVELS,
    MODEL_REGISTRY,
    OPENAI_BASE_URL,
    OPENROUTER_BASE_URL,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    PROVIDER_OPENROUTER,
    THINKING_ADAPTIVE,
    THINKING_BUDGET,
    THINKING_DISABLED,
    THINKING_DISPLAY_OMITTED,
    THINKING_DISPLAY_SUMMARIZED,
    THINKING_DISPLAYS,
    THINKING_MODES,
    WIRE_CHAT_COMPLETIONS,
    WIRE_RESPONSES,
    Model,
    ThinkingSupport,
    is_known_model,
    is_retired,
    known_models,
    model_info,
    model_supports_images,
    supports_forced_tool_choice,
    SAMPLING_PARAMS,
    rejected_sampling_params,
    sampling_band,
    thinking_support,
)
from direktoro.routing import (
    GATEWAY_OPENROUTER,
    ProviderRouteMismatch,
    Route,
    call_identity_fields,
    canonical_json,
    fingerprint_fields,
)
from direktoro.batch import (
    BatchResultError,
    build_batch_client,
    map_batch_results,
    normalise_batch_message,
    run_message_batch,
)
from direktoro.wire_log import (
    redact_messages,
    redact_system,
    redact_wire_request,
    response_to_dict,
)

# The single source of truth for the package version: `pyproject.toml` declares
# `dynamic = ["version"]` and reads this attribute at build time
# (`[tool.setuptools.dynamic] version = { attr = "direktoro.__version__" }`), so
# the distribution metadata and `direktoro.__version__` cannot drift. Bump it
# here and nowhere else: this version can end up inside a caller's own
# call-identity fingerprint, so a bump moves published provenance and must be
# deliberate. `tests/test_public_api.py` asserts the wiring.
__version__ = "0.3.0"

__all__ = [
    "__version__",
    # Adapter construction + call
    "build_adapter",
    "create_message_with_retry",
    "MissingAPIKey",
    # Forced-named-tool helpers
    "tool_choice_named",
    "extract_tool_call",
    # Audit-log helpers (direktoro.wire_log): stub inline image bytes out of
    # a request before logging it, and flatten a provider SDK response object
    # to a plain dict. Here because which block shapes carry image bytes on
    # which wire is this package's knowledge, not a consumer's.
    "redact_messages",
    "redact_system",
    "redact_wire_request",
    "response_to_dict",
    # Decoding params (single source of truth), and the splitter that takes
    # one role's decoding block from an application config — sampling controls
    # and thinking_* fields side by side — and returns the (sampling,
    # thinking) pair the resolver takes, so the application carries the block
    # without knowing which key is which.
    "resolved_decoding_params",
    "split_decoding_config",
    # Thinking / reasoning-effort seam. `Thinking` is the per-call request
    # spec; `ThinkingSupport` is the registry's per-model capability record;
    # `ThinkingUnsupported` is raised BEFORE the call when the registry knows
    # the endpoint would 400 on the requested shape. A chosen mode or effort
    # reaches call identity through `resolved_decoding_params`, so two runs
    # differing only in effort fingerprint differently.
    "Thinking",
    "ThinkingSupport",
    "ThinkingUnsupported",
    "thinking_support",
    "EFFORT_LEVELS",
    "THINKING_ADAPTIVE",
    "THINKING_DISABLED",
    "THINKING_BUDGET",
    "THINKING_MODES",
    "THINKING_DISPLAYS",
    "THINKING_DISPLAY_SUMMARIZED",
    "THINKING_DISPLAY_OMITTED",
    # Normalised exceptions
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderRetryableError",
    # Normalised response
    "NormalisedResponse",
    "NormalisedUsage",
    "RETRY_BACKOFF_SECONDS",
    # Adapter classes, for a caller that wraps a client itself rather than
    # going through `build_adapter`.
    "AnthropicAdapter",
    "OpenAIAdapter",
    # Model registry: how to reach a model and what it can do. Rates are their
    # own seam — see `PRICES` and `cost_from_rates`.
    "MODEL_REGISTRY",
    "model_info",
    "Model",
    "is_known_model",
    "known_models",
    "model_supports_images",
    # The new-run gate, for callers that have one. Nothing here refuses a
    # retired id on your behalf — every lookup keeps resolving one so the
    # provenance of a past run survives — so a consumer applies `is_retired`
    # (or `known_models(include_retired=False)`) wherever it accepts a new run.
    "is_retired",
    # Cost arithmetic over the CALLER's rates (USD per million tokens, keyed by
    # counter). A non-zero counter with no rate raises rather than costing zero,
    # so a total can never be quietly too low. Knows nothing about any model or
    # provider, which is what keeps it from going stale. `cost_from_usage`
    # prices a whole `NormalisedUsage` — every counter, mapped once here rather
    # than at each call site, since the counter a hand-written mapping omits is
    # the one that then costs nothing at all.
    "cost_from_rates",
    "cost_from_usage",
    # The dated price table: each entry is one vendor's published per-million
    # rates, the page they were read from and the day they were read.
    # `price_for(model_id).as_rates()` is the mapping `cost_from_rates` takes;
    # `price_age_days(model_id, today)` measures how old the reading is against
    # a date the caller passes; `PRICES_VERSION` identifies the table's data, so
    # a run can record which table priced it. The table covers the direct
    # models — a routed call carries the gateway's own charge on the response.
    "PriceEntry",
    "PRICES",
    "PRICES_VERSION",
    "price_for",
    "is_priced",
    "price_age_days",
    # Engine provenance: a sha256 over this package's own source files, for a
    # consumer that records which bytes produced a run alongside which release.
    "source_hash",
    # Capability predicate: does the model's endpoint honour a forced named
    # tool_choice? Stated per entry, never defaulted; a consumer arms its
    # auto-degrade retry off a False.
    "supports_forced_tool_choice",
    # Capability predicate: does the model's endpoint accept the sampling
    # controls (temperature/top_p)? False for google/gemini-3.6-flash, whose
    # Vertex endpoints dropped them; resolved_decoding_params omits an
    # unaccepted sampling param from wire AND fingerprint.
    "SAMPLING_PARAMS",
    "rejected_sampling_params",
    # Capability lookup: the (low, high) range documented for one sampling
    # param — the model's own reference for a direct entry, its gateway's
    # request surface for a routed one — or None where none is established.
    "sampling_band",
    # Provider / wire constants. Three providers, which is every value a
    # registry entry can carry and every value `call_identity_fields` can
    # report. The two pinned base URLs are here for the same reason the
    # provider strings are: `call_identity_fields` reports a `base_url`, and a
    # consumer checking which endpoint an identity block describes should
    # compare against a named constant rather than retype the URL or reach into
    # `direktoro.registry` for it. (Anthropic pins none — it uses the SDK
    # default — so its identity blocks carry `base_url: None` and there is no
    # constant to export.)
    "PROVIDER_ANTHROPIC",
    "PROVIDER_OPENAI",
    "PROVIDER_OPENROUTER",
    "OPENAI_BASE_URL",
    "OPENROUTER_BASE_URL",
    "WIRE_CHAT_COMPLETIONS",
    "WIRE_RESPONSES",
    # Routing & provenance: Route, canonical serialisers for consumer
    # fingerprints, the centralised provider-call identity block, and the
    # pin-mismatch exception.
    "Route",
    "GATEWAY_OPENROUTER",
    "ProviderRouteMismatch",
    "fingerprint_fields",
    "call_identity_fields",
    "canonical_json",
    # Anthropic Message Batches
    "run_message_batch",
    "map_batch_results",
    "normalise_batch_message",
    "build_batch_client",
    "BatchResultError",
]
