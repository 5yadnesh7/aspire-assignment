from pathlib import Path

from aspire_eval.client import assert_cloud_host
from aspire_eval.config import load_config
from aspire_eval.cost import estimate_cost

import pytest


ROOT = Path(__file__).parents[1]


def test_config_and_official_prices_load() -> None:
    _, config = load_config(ROOT)
    assert config.provider.base_url == "https://ollama.com"
    assert config.models["primary_judge"].model == "gpt-oss:120b"
    assert config.models["primary_judge"].pricing_usd_per_1m_tokens.input == 0.15
    assert config.models["independent_critic"].model == "gemma4:31b"
    assert config.models["independent_critic"].pricing_usd_per_1m_tokens.output == 0.4


def test_cost_uses_cached_and_uncached_rates() -> None:
    _, config = load_config(ROOT)
    model = config.models["primary_judge"]
    estimate = estimate_cost(
        {
            "prompt_eval_count": 1_000_000,
            "prompt_eval_cached_count": 250_000,
            "eval_count": 1_000_000,
        },
        model,
        config.usage_fields,
    )
    assert estimate.usd == pytest.approx(0.716)
    assert estimate.cached_tokens_known is True


def test_missing_cached_tokens_use_uncached_rate_and_are_labeled() -> None:
    _, config = load_config(ROOT)
    estimate = estimate_cost(
        {"prompt_eval_count": 1000, "eval_count": 0},
        config.models["primary_judge"],
        config.usage_fields,
    )
    assert estimate.usd == pytest.approx(0.00015)
    assert estimate.cached_tokens_known is False
    assert estimate.assumption is not None


def test_missing_usage_does_not_invent_a_zero_cost() -> None:
    _, config = load_config(ROOT)
    estimate = estimate_cost({}, config.models["primary_judge"], config.usage_fields)
    assert estimate.usd is None
    assert estimate.known is False


def test_localhost_is_rejected() -> None:
    with pytest.raises(RuntimeError):
        assert_cloud_host("http://localhost:11434")
    with pytest.raises(RuntimeError):
        assert_cloud_host("https://127.0.0.1")
