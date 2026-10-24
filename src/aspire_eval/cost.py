from dataclasses import dataclass

from aspire_eval.config import ModelConfig, UsageFields


@dataclass(frozen=True)
class CostEstimate:
    usd: float | None
    input_tokens: int | None
    cached_input_tokens: int | None
    uncached_input_tokens: int | None
    output_tokens: int | None
    cached_tokens_known: bool
    known: bool
    assumption: str | None = None


def estimate_cost(response: dict, pricing: ModelConfig, fields: UsageFields) -> CostEstimate:
    input_tokens = _optional_int(response.get(fields.input_tokens))
    output_tokens = _optional_int(response.get(fields.output_tokens))
    if input_tokens is None or output_tokens is None:
        return CostEstimate(
            usd=None,
            input_tokens=input_tokens,
            cached_input_tokens=None,
            uncached_input_tokens=None,
            output_tokens=output_tokens,
            cached_tokens_known=False,
            known=False,
            assumption="Token counts were absent from the provider response",
        )

    cached_raw = response.get(fields.cached_input_tokens)
    cached_known = cached_raw is not None
    cached_tokens = _optional_int(cached_raw) or 0
    uncached_tokens = max(input_tokens - cached_tokens, 0)
    rates = pricing.pricing_usd_per_1m_tokens
    usd = (
        uncached_tokens * rates.input
        + cached_tokens * rates.cached_input
        + output_tokens * rates.output
    ) / 1_000_000
    assumption = None
    if not cached_known:
        assumption = "Cached input tokens were not reported; all input tokens use the uncached rate"
    return CostEstimate(
        usd=usd,
        input_tokens=input_tokens,
        cached_input_tokens=cached_tokens if cached_known else None,
        uncached_input_tokens=uncached_tokens,
        output_tokens=output_tokens,
        cached_tokens_known=cached_known,
        known=True,
        assumption=assumption,
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
