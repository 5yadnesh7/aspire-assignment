import json

from aspire_eval.parsing import extract_json


def test_extracts_fenced_json_after_preamble() -> None:
    text = 'Here is the result:\n```json\n{"evaluations": [{"test_case_id": "tc-1"}]}\n```'
    assert extract_json(text)["evaluations"][0]["test_case_id"] == "tc-1"


def test_ignores_earlier_json_without_expected_key() -> None:
    text = '{"example": true}\n{"reviews": [{"test_case_id": "tc-2"}]}'
    assert extract_json(text)["reviews"][0]["test_case_id"] == "tc-2"
