# Wallet top-up test evaluator

This evaluates the four generated wallet top-up tests from the Aspire Quality Platform take-home. It calls Ollama Cloud, validates both model outputs with Pydantic, and calculates the final scores in Python.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m aspire_eval --preflight-only
.\.venv\Scripts\python -m aspire_eval --show
.\.venv\Scripts\python -m aspire_eval
```

Put `OLLAMA_API_KEY` in `.env`. The client rejects localhost and any non-HTTPS host.

## Models

- Primary judge: `gpt-oss:120b`
- Independent critic: `gemma4:31b`

`kimi-k2.7-code` was the first critic choice. The Ollama account returned HTTP 402 for it because that model is outside the current plan, and named `gemma4` as an available alternative. The submitted runs use `gemma4:31b`.

Prices, temperature, weights, and the spend ceiling live in `config/models.json`.

## How a score is produced

1. The primary model labels every atomic requirement for every test and gives two dimension scores.
2. The critic may downgrade those labels or scores. Upgrades are ignored.
3. Deterministic checks cap claims that contradict the text, including the five-second timeout, the SGD 10,000 boundary, second-owner approval, and credit-before-debit ordering.
4. A `covered` label whose evidence is not an exact quote is downgraded.
5. Python calculates the focused quality score and the suite gate score. The model does not choose the final number.

The suite gate is the score used for the CI threshold. A critical requirement that remains missing or incorrect blocks a merge even when the numeric score passes.

## Outputs

- `outputs/runs/run-01` through `run-05`: prompts, raw provider responses, validation errors, and reconciled results
- `outputs/runs/adversarial-01`: a spec paraphrase whose hidden implementation asserts nothing
- `outputs/events.jsonl`: one observability record per model call
- `outputs/summary.json`: variance, agreement, recommended threshold, and cost

Ollama Cloud does not constrain output to a JSON schema. The system prompt requests the schema, and Pydantic rejects anything else. One correction retry is allowed. Raw failed responses are retained.
