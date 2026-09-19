import pytest

from aqar_intel.llm import FakeLLM, MockLLM, extract_json


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure! ```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"sql": "SELECT 1"} thanks') == {"sql": "SELECT 1"}
    assert extract_json("[1, 2]") == [1, 2]


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_mock_and_fake():
    assert MockLLM().chat([{"role": "user", "content": "x"}], json_mode=True) == "{}"
    f = FakeLLM(["a", "b"])
    assert f.chat([]) == "a" and f.chat([]) == "b"
    with pytest.raises(AssertionError):
        f.chat([])
