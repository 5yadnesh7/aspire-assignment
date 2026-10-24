import json
import re

from aspire_eval.schemas import strip_model_wrappers


def message_content(message: dict) -> tuple[str, bool]:
    content = message.get("content") or ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        content = "\n".join(parts)
    text = str(content).strip()
    if text:
        return text, False
    thinking = str(message.get("thinking") or "").strip()
    return thinking, bool(thinking)


def extract_json(text: str) -> dict:
    cleaned = strip_model_wrappers(text)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    candidates: list[dict] = []
    search_text = fenced.group(1).strip() if fenced else cleaned
    candidates.extend(_json_objects(search_text))
    if not candidates:
        candidates.extend(_json_objects(cleaned))
    preferred = [
        item for item in candidates if "evaluations" in item or "reviews" in item
    ]
    if preferred:
        return preferred[-1]
    if candidates:
        return candidates[-1]
    raise ValueError("Model response did not contain a JSON object")


def _json_objects(text: str) -> list[dict]:
    stripped = text.strip()
    found: list[dict] = []
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return [value]
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            found.append(value)
    return found
