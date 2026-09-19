from types import SimpleNamespace

import pytest

from aqar_intel.config import Settings
from aqar_intel.llm import FakeLLM, MockLLM, OpenRouterLLM, extract_json


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


def _resp(content: str, finish_reason: str = "stop"):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=content))])


class _StubCompletions:
    """Stands in for ``openai.OpenAI().chat.completions``; records kwargs and replays scripted responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_openrouter_adds_reasoning_headroom_and_retries_truncated_replies():
    # Reasoning models spend hidden thinking tokens from max_tokens: a 50-token router call comes back as '{"route'.
    settings = Settings(openrouter_api_key="sk-test", llm_mode="openrouter", reasoning_headroom=1000, reasoning_effort="low")
    llm = OpenRouterLLM(settings)
    stub = _StubCompletions([_resp('{"route', "length"), _resp('{"route": "docs"}')])
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=stub))

    out = llm.chat([{"role": "user", "content": "q"}], json_mode=True, max_tokens=50)

    assert out == '{"route": "docs"}'
    assert stub.calls[0]["max_tokens"] == 1050
    assert stub.calls[0]["response_format"] == {"type": "json_object"}
    assert stub.calls[0]["extra_body"] == {"reasoning": {"effort": "low"}}
    assert stub.calls[1]["max_tokens"] == 3150  # grown after finish_reason == "length"


def test_openrouter_drops_reasoning_param_if_provider_rejects_it():
    settings = Settings(openrouter_api_key="sk-test", llm_mode="openrouter", reasoning_effort="low")

    class _Rejecting(_StubCompletions):
        def create(self, **kwargs):
            if "extra_body" in kwargs:
                self.calls.append(kwargs)
                raise RuntimeError("400: reasoning is not supported by this model")
            return super().create(**kwargs)

    llm = OpenRouterLLM(settings)
    stub = _Rejecting([_resp("ok")])
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=stub))
    assert llm.chat([{"role": "user", "content": "q"}]) == "ok"
    assert "extra_body" not in stub.calls[-1]
