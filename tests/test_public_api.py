"""The direktoro public API.

`direktoro/__init__.py`'s `__all__` is the public surface, and this module
holds it to two properties:

  1. Every name below is importable from the package ROOT. The lists here are a
     deliberate second copy of `__all__` — a test that read `__all__` and then
     checked `__all__` would pass no matter what was removed from it, so the
     names are restated, grouped by seam, and a name dropped from the package
     fails here.
  2. `import direktoro` succeeds without the `anthropic`/`openai` SDKs
     installed: both are imported lazily inside the adapters, never at package
     import time, so a consumer that installs the wheel `--no-deps` can still
     `import direktoro`, read the registry, and cost usage with
     `cost_from_rates`.

Plus the packaging properties a consumer depends on but cannot see from the
installed package: one source of truth for the version, a distribution that
ships a runnable suite, and dependency floors that are measured rather than
guessed.
"""

import importlib
import re
import sys
from pathlib import Path

import pytest


# The core public surface: what a consumer needs to build an adapter, make a
# call, read the registry, and cost the usage that comes back.
EXPECTED_PUBLIC_API = [
    "build_adapter",
    "create_message_with_retry",
    "extract_tool_call",
    "resolved_decoding_params",
    "tool_choice_named",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderRetryableError",
    "MissingAPIKey",
    "NormalisedResponse",
    "NormalisedUsage",
    "RETRY_BACKOFF_SECONDS",
    "MODEL_REGISTRY",
    "model_info",
    "cost_from_rates",
    "cost_from_usage",
    "Model",
]


def test_every_public_name_is_a_top_level_attribute():
    import direktoro

    missing = [name for name in EXPECTED_PUBLIC_API
               if not hasattr(direktoro, name)]
    assert missing == []


def test_every_public_name_is_exported_in_all():
    import direktoro

    missing = [name for name in EXPECTED_PUBLIC_API
               if name not in direktoro.__all__]
    assert missing == []


# The rest of the surface, grouped by seam. Consumers import ONLY from the
# package root, so every consumer-facing name must be a top-level attribute AND
# in __all__.
ROUTING_NAMES = [
    "Route",
    "GATEWAY_OPENROUTER",
    "ProviderRouteMismatch",
    "fingerprint_fields",
    "call_identity_fields",
    "canonical_json",
    "PROVIDER_OPENROUTER",
]


def test_routing_names_are_public():
    import direktoro

    for name in ROUTING_NAMES:
        assert hasattr(direktoro, name), name
        assert name in direktoro.__all__, name


# The capability lookups a consumer branches on: whether the endpoint honours
# a forced named tool_choice (stated per entry, never defaulted; a False arms
# the consumer's auto-degrade retry), which sampling controls it refuses, and
# the documented value range for one it accepts.
CAPABILITY_PREDICATES = ["supports_forced_tool_choice",
                         "rejected_sampling_params",
                         "sampling_band"]


def test_capability_predicates_are_public():
    import direktoro

    for name in CAPABILITY_PREDICATES:
        assert hasattr(direktoro, name), name
        assert name in direktoro.__all__, name


# The price seam: the dated table, its lookups, and the version stamp a run
# records to say which table priced it. A consumer imports these from the root
# alongside `cost_from_rates`, which takes what `as_rates()` produces.
PRICE_NAMES = [
    "PriceEntry",
    "PRICES",
    "PRICES_VERSION",
    "price_for",
    "is_priced",
    "price_age_days",
]


def test_price_names_are_public():
    import direktoro

    for name in PRICE_NAMES:
        assert hasattr(direktoro, name), name
        assert name in direktoro.__all__, name


def test_the_price_table_reaches_the_arithmetic_from_the_root():
    # Usable, not merely importable: an entry's rates go straight into
    # `cost_from_rates` without the caller rekeying anything.
    import direktoro

    rates = direktoro.price_for("claude-opus-5").as_rates()
    assert direktoro.cost_from_rates(
        rates=rates, input_tokens=1_000_000) == pytest.approx(5.00)


def test_source_hash_is_public_and_answers():
    """`source_hash()` names the bytes of the running copy, for a consumer that
    folds engine identity into a run record. It has to be reachable from the
    root like everything else a consumer stores."""
    import direktoro

    assert "source_hash" in direktoro.__all__
    digest = direktoro.source_hash()
    assert digest == direktoro.source_hash()
    assert re.fullmatch(r"[0-9a-f]{64}", digest) or digest == "nosource"


# The retirement gate. Nothing in this package refuses a retired id on a
# caller's behalf — every lookup keeps resolving one so a past run stays
# resolvable and citable — so a library consumer needs a way to ask before it
# starts a new run. Without these two names on the public surface, the CLI has a
# gate and everybody importing the package has none.
RETIREMENT_GATE_NAMES = ["is_retired", "known_models"]


def test_the_retirement_gate_is_importable_from_the_root():
    import direktoro

    for name in RETIREMENT_GATE_NAMES:
        assert hasattr(direktoro, name), name
        assert name in direktoro.__all__, name


def test_the_retirement_gate_answers_from_the_package_root():
    # Not just importable: usable, in both its forms, without reaching into a
    # submodule or reading `Model.retired` off the record by hand.
    import direktoro

    retired = [model_id for model_id, info in direktoro.MODEL_REGISTRY.items()
               if info.retired]
    assert retired, "no retired entry to check the gate against"
    for model_id in retired:
        assert direktoro.is_retired(model_id) is True
        assert model_id not in direktoro.known_models(include_retired=False)
        # Still resolvable, which is the whole reason the gate has to be
        # applied rather than assumed.
        assert direktoro.model_info(model_id) is not None


def test_the_pinned_base_urls_are_public_and_match_the_table():
    """`call_identity_fields` reports a `base_url`, so a consumer checking which
    endpoint a stored identity block describes needs the same constant the table
    was built from. Exported so that check is a comparison against a name rather
    than a retyped URL or a reach into `direktoro.registry`, and held equal to
    what the entries actually carry so the two cannot drift apart."""
    import direktoro

    for name in ("OPENAI_BASE_URL", "OPENROUTER_BASE_URL"):
        assert hasattr(direktoro, name), name
        assert name in direktoro.__all__, name

    by_provider = {}
    for info in direktoro.MODEL_REGISTRY.values():
        by_provider.setdefault(info.provider, set()).add(info.base_url)
    assert by_provider["openai"] == {direktoro.OPENAI_BASE_URL}
    assert by_provider["openrouter"] == {direktoro.OPENROUTER_BASE_URL}
    # Anthropic pins none (the SDK default), which is why no constant exists for
    # it and why its identity blocks carry base_url None.
    assert by_provider["anthropic"] == {None}


def test_every_exported_provider_constant_is_one_the_registry_uses():
    """A public constant naming a provider class this package does not have is
    a promise it cannot keep: a consumer branching on it writes a branch nothing
    will ever take, and a reader of `call_identity_fields` expects a value that
    can never appear there. So the exported set and the set the table actually
    produces are held equal in both directions — an unused constant fails here
    just as loudly as a missing one."""
    import direktoro

    exported = {getattr(direktoro, name) for name in direktoro.__all__
                if name.startswith("PROVIDER_")}
    used = {info.provider for info in direktoro.MODEL_REGISTRY.values()}
    assert exported == used, (
        "the exported PROVIDER_* constants and the provider values the "
        "registry actually carries have diverged. An exported constant no "
        "entry uses is a public promise about a provider class this package "
        "does not have; drop it, or add the entries that justify it.")


# The thinking / reasoning-effort seam. `Thinking` is the per-call request
# spec; `ThinkingSupport` is the registry's per-model capability record;
# `ThinkingUnsupported` is raised before the call for a shape the endpoint would
# 400 on. All optional: a consumer that imports none of them is unaffected.
THINKING_SEAM_NAMES = [
    "Thinking",
    "ThinkingSupport",
    "ThinkingUnsupported",
    "thinking_support",
    "EFFORT_LEVELS",
    "THINKING_ADAPTIVE",
    "THINKING_DISABLED",
    "THINKING_BUDGET",
    "THINKING_MODES",
]


def test_thinking_seam_names_are_public():
    import direktoro

    for name in THINKING_SEAM_NAMES:
        assert hasattr(direktoro, name), name
        assert name in direktoro.__all__, name


def test_from_import_of_the_full_surface_binds_every_name():
    # A real from-import of the whole list: fails at collection if any name is
    # unbound, so this is the byte-compatibility assertion the consumers rely on.
    from direktoro import (  # noqa: F401
        MODEL_REGISTRY,
        MissingAPIKey,
        Model,
        NormalisedResponse,
        NormalisedUsage,
        ProviderError,
        ProviderRateLimitError,
        ProviderRetryableError,
        RETRY_BACKOFF_SECONDS,
        build_adapter,
        cost_from_rates,
        cost_from_usage,
        create_message_with_retry,
        extract_tool_call,
        model_info,
        resolved_decoding_params,
        tool_choice_named,
    )


def test_import_succeeds_without_the_sdks(monkeypatch):
    # Simulate the SDKs being absent: drop any cached copies of direktoro and
    # the two SDKs, then poison `anthropic`/`openai` in sys.modules so importing
    # either raises ImportError. `import direktoro` must still succeed and expose
    # the full public surface, because the SDK imports live inside the adapters
    # (and build_adapter), not at module top level.
    for name in list(sys.modules):
        if (name == "direktoro" or name.startswith("direktoro.")
                or name == "anthropic" or name.startswith("anthropic.")
                or name == "openai" or name.startswith("openai.")):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "anthropic", None)
    monkeypatch.setitem(sys.modules, "openai", None)

    # The SDKs are genuinely unavailable now (None in sys.modules raises).
    with pytest.raises(ImportError):
        importlib.import_module("anthropic")
    with pytest.raises(ImportError):
        importlib.import_module("openai")

    # The package imports and every public name is present, no SDK required.
    direktoro = importlib.import_module("direktoro")
    missing = [name for name in EXPECTED_PUBLIC_API
               if not hasattr(direktoro, name)]
    assert missing == []
    # And the SDK-free paths actually work: the registry resolves an id, and
    # the cost arithmetic runs on the caller's rates.
    assert direktoro.model_info("claude-opus-4-7").provider == "anthropic"
    assert direktoro.cost_from_rates(
        rates={"input": 5.0, "output": 25.0},
        input_tokens=1_000_000, output_tokens=1_000_000) \
        == pytest.approx(30.00)


# ---------------------------------------------------------------------------
# Version: one source of truth
# ---------------------------------------------------------------------------

def _pyproject():
    """The repo's pyproject.toml, or None when running outside a source tree."""
    import tomllib

    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not path.is_file():
        return None
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_version_has_exactly_one_source_of_truth():
    """`pyproject.toml` must READ the version from `direktoro.__version__`
    rather than restate it. Two literals are a drift waiting to happen: a bump
    in one place and not the other ships a wheel whose metadata disagrees with
    its own `__version__`, and this version can end up inside a caller's
    call-identity fingerprint, where a disagreement is not visible at all."""
    config = _pyproject()
    if config is None:
        pytest.skip("not running from a source checkout")
    project = config["project"]
    assert "version" not in project, (
        "pyproject.toml declares a literal version as well as reading it from "
        "the package; that is the drift this test exists to prevent.")
    assert "version" in project.get("dynamic", [])
    assert config["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "direktoro.__version__"}


def test_version_is_a_real_release_not_a_dev_placeholder():
    """A `.dev0` / `a` / `b` / `rc` suffix makes the version a pre-release, and
    `pip install direktoro` skips a pre-release unless the caller passes
    `--pre`. A published package whose only release is invisible to a plain
    install is not installable in practice."""
    import direktoro

    version = direktoro.__version__
    assert not any(marker in version
                   for marker in (".dev", "a", "b", "rc")), version
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), version


def _manifest_rules():
    """The repo's MANIFEST.in as a list of rules, or None outside a checkout."""
    path = Path(__file__).resolve().parents[1] / "MANIFEST.in"
    if not path.is_file():
        return None
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def test_the_sdist_ships_the_whole_test_suite_not_a_slice_of_it():
    """setuptools' default sdist sweeps up `tests/test*.py` and nothing else
    under `tests/`, which leaves out `tests/conftest.py` — the file holding the
    session-wide network guard. A shipped suite without its guard is worse than
    no shipped suite: the guard is what makes "these tests cannot reach the
    network and cannot spend money" checkable, and the tests go looking for real
    hosts without it. CI cannot notice on its own, because CI runs pytest from
    the checkout, where conftest.py is always present.

    MANIFEST.in is what makes the shipped suite complete; this asserts the
    declaration is still there. The full check is the `sdist-suite` CI job:
    build, unpack somewhere clean with nothing but pytest, and run what was
    shipped.
    """
    rules = _manifest_rules()
    if rules is None:
        pytest.skip("not running from a source checkout")
    assert "recursive-include tests *.py" in rules, (
        "MANIFEST.in must recursively include tests/*.py, or the sdist ships "
        "test modules without their conftest.py and the shipped suite goes to "
        "the network.")
    # Build droppings vary between machines, so two builds of one commit would
    # otherwise differ, and a shipped stale .pyc can shadow its own source.
    assert "global-exclude *.py[cod]" in rules
    assert "global-exclude .DS_Store" in rules
    assert "prune **/__pycache__" in rules


def test_the_shipped_suite_can_find_the_package_without_an_install():
    """An unpacked sdist has `src/` but no installed `direktoro`, so without a
    pytest `pythonpath` every shipped test errors on `import direktoro` and the
    complete suite is unrunnable — not much better than the broken one."""
    config = _pyproject()
    if config is None:
        pytest.skip("not running from a source checkout")
    assert config["tool"]["pytest"]["ini_options"]["pythonpath"] == ["src"]


def test_the_installed_sdk_accepts_what_the_adapter_sends():
    """The live half of the `anthropic` floor.

    The floor exists because `AnthropicAdapter.create_message` passes
    `thinking=` and `output_config=` to `messages.stream(...)`, which are named
    parameters with no `**kwargs` — an SDK missing either raises TypeError at
    CALL time, not at install time, so nothing catches it until a paid run does.
    The declared floor (>=0.77, the measured first release carrying both) is a
    claim about published wheels this repo cannot re-check offline; THIS checks
    the one SDK that is actually going to be called."""
    anthropic = pytest.importorskip("anthropic")
    import inspect

    from anthropic.resources.messages import Messages

    signature = inspect.signature(Messages.stream)
    for parameter in ("thinking", "output_config"):
        assert parameter in signature.parameters, (
            f"the installed anthropic SDK ({anthropic.__version__}) has no "
            f"`{parameter}` parameter on messages.stream, so every call "
            f"carrying a thinking spec would TypeError. Raise the floor in "
            f"pyproject.toml, or upgrade the SDK.")


def test_the_declared_floors_are_the_justified_ones():
    """A dependency floor is a claim about which released versions work, and an
    unexplained number is an unfalsifiable one. A floor set to whatever happened
    to be installed on the day over-constrains every consumer downstream, and
    nothing in a normal test run notices. So both floors are pinned here and
    justified in a comment beside them in pyproject.toml: moving one is then a
    deliberate act with a justification to rewrite.

      - `anthropic>=0.77`: `messages.stream(...)` gained `output_config` there
        (and `thinking` in 0.47.0); the adapter passes both as named parameters.
      - `openai>=1.66`: `client.responses.create` — the Responses API — first
        ships in openai-python 1.66.0, and the OpenAI adapter's direct path
        calls it.
    """
    config = _pyproject()
    if config is None:
        pytest.skip("not running from a source checkout")
    floors = {requirement.split(">=")[0].strip(): requirement
              for requirement in config["project"]["dependencies"]}
    assert floors["anthropic"] == "anthropic>=0.77", (
        "the anthropic floor moved; update the measurement recorded beside it "
        "in pyproject.toml, or this number is unexplained again.")
    assert floors["openai"] == "openai>=1.66", (
        "the openai floor moved; update the justification recorded beside it "
        "in pyproject.toml, or this number is unexplained again.")


def test_the_openai_floor_matches_what_the_adapter_calls():
    """The live half of the `openai` floor, mirroring the `anthropic` one above.

    `OpenAIAdapter.create_message` reaches `client.responses.create(...)` on the
    direct-OpenAI path. `responses` is the Responses API namespace, added in
    openai-python 1.66.0 — an older SDK has no `responses` attribute at all, so
    the failure is an AttributeError at CALL time, not at install time. The
    declared floor is a claim about published wheels this repo cannot re-check
    offline; this checks the SDK that is actually going to be called.

    ASSERTED ON AN INSTANCE, WHICH IS THE ONLY FORM THAT MATCHES THE CLAIM. The
    SDK has bound its resource namespaces two ways over the range this floor
    covers: assigned in `Client.__init__` (`self.responses = ...`) in the older
    releases, and declared as a class-level `cached_property` in the newer ones.
    An instance lookup finds it under both — an attribute assigned in `__init__`
    and a descriptor on the class resolve identically from an instance — while a
    CLASS-level check sees only the second and reports the floor release as
    lacking a namespace it has. That check would pass in CI, where the newest
    SDK is installed, and fail for the one consumer who pinned the floor this
    test is about, telling them to raise a floor that is correct. The instance
    is also what the adapter actually holds.

    No network: constructing a client builds no connection and resolves no name,
    and the suite-wide guard in conftest.py would fail this test if it did.
    """
    openai = pytest.importorskip("openai")

    client = openai.OpenAI(api_key="not-a-real-key")
    assert client.responses is not None, (
        f"the installed openai SDK ({openai.__version__}) exposes no "
        f"`responses` namespace, so every direct OpenAI call would fail at "
        f"call time. Raise the floor in pyproject.toml, or upgrade the SDK.")
