"""The root of the normalised provider-failure hierarchy.

`ProviderError` is what every provider failure this package raises inherits
from, so one `except ProviderError` catches them all whatever SDK or wire
produced the failure — including the routing refusals, which are provider
failures like any other.

IT SITS ALONE IN A MODULE THAT IMPORTS NOTHING, because two modules raise from
this tree and neither can import the other. `direktoro.providers` translates the
SDK exceptions and imports `direktoro.routing` for the pin assertion;
`direktoro.routing` raises `ProviderRouteMismatch` and is imported by
`direktoro.registry`, which `direktoro.providers` imports in turn. A leaf module
both can import is what keeps the hierarchy single-rooted without a cycle.

The subclasses live beside the code that raises them — `ProviderRateLimitError`
and `ProviderRetryableError` in `direktoro.providers`, `ProviderRouteMismatch`
in `direktoro.routing` — and every one of them, this base included, is imported
from the package root by consumers.
"""


class ProviderError(Exception):
    """A provider API error, normalised across SDKs.

    The concrete SDK exception (anthropic.APIError, openai.APIError, ...) is
    translated into this or a subclass so callers' retry logic is provider
    independent.

    `response` IS THE BILLED MATERIAL THE REFUSAL IS ABOUT, and it is None on
    every failure raised INSTEAD of a response — a call that never reached the
    provider, or reached it and got an error back, has nothing to carry. It is
    set on the other kind: a call that WAS SERVED AND BILLED whose result this
    layer refuses anyway (a routed response whose pin, audit receipt or cost
    figure does not hold up). The money is already spent by then, so the
    `NormalisedResponse` as it stood — content, usage, and whatever routing
    fields were established before the refusal — rides on the exception instead
    of being discarded with it, and a consumer catching the refusal can still
    ledger the tokens and any reported cost against the run.

    `provider_message` IS THE PROVIDER'S OWN SENTENCE, lifted out of the error
    body and set aside from the message. It is what a human is told to DO —
    "You have no credits remaining. Add credits to continue using the API at
    ..." — in the words of the party that can act on it, and a consumer showing
    a failure to an operator wants exactly that and none of the wrapper around
    it. `str(error)` still carries everything the SDK built, envelope
    included, so nothing is lost by reading this instead; it is the same text
    with the machinery taken off.

    It is None on a failure NO PROVIDER SPOKE ON: the routing refusals this
    package raises itself (a pin mismatch, a missing receipt, a missing cost)
    are direktoro's reasoning about a response that arrived fine, and inventing
    a provider sentence for them would misattribute it. `error.provider_message
    or str(error)` is the line a consumer wants — the clean sentence where
    there is one, the full text where there is not.

    IT IS SET STRUCTURALLY, off the parsed body, never by reading `str(error)`.
    That is the point of it: a consumer that would otherwise pick the sentence
    out of the SDK's rendering is one provider rewording away from showing an
    operator nothing.
    """

    response = None
    provider_message = None
