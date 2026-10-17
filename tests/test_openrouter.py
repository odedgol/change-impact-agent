"""OpenRouter adapter: same OpenAI-compatible mapping as Ollama, hosted, with a key and rate-limit retries. No network."""

import pytest

from cia.llm import (
    DEFAULT_OPENROUTER_MODEL,
    BedrockClient,
    HTTPStatusError,
    LLMClient,
    OpenAICompatClient,
    OpenRouterClient,
    UserText,
    make_client,
    retrying,
)
from cia.monitor import Monitor

OK = {
    "choices": [
        {"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    ]
}


class Stub:
    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []

    def __call__(self, url, body, headers):
        self.requests.append({"url": url, "body": body, "headers": headers})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    for k in ("OPENROUTER_API_KEY", "CIA_OPENROUTER_MODEL", "CIA_OPENAI_MAX_TOKENS"):
        monkeypatch.delenv(k, raising=False)


def test_is_an_llm_client_next_to_ollama():
    client = OpenRouterClient(api_key="k", transport=Stub(OK))
    assert isinstance(client, LLMClient) and isinstance(client, OpenAICompatClient)
    assert client.model == DEFAULT_OPENROUTER_MODEL == "google/gemma-4-31b-it:free"


def test_needs_a_key():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterClient()


def test_request_goes_to_openrouter_with_key_and_title(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("CIA_OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
    stub = Stub(OK)
    turn = OpenRouterClient(transport=stub).converse("sys", [UserText("hi")], [])
    req = stub.requests[0]
    assert req["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert (
        req["headers"]["Authorization"] == "Bearer sk-or-test"
        and req["headers"]["X-Title"]
    )
    assert req["body"]["model"] == "google/gemma-4-26b-a4b-it:free"
    assert turn.text == "hi"


def test_ollama_client_is_unchanged():
    stub = Stub(OK)
    OpenAICompatClient(model="gemma4:26b-mlx", transport=stub).converse(
        "sys", [UserText("hi")], []
    )
    assert stub.requests[0]["url"] == "http://localhost:11434/v1/chat/completions"
    assert "X-Title" not in stub.requests[0]["headers"]


def test_rate_limits_are_retried_then_succeed():
    waits = []
    stub = Stub(
        HTTPStatusError(429, "slow down", retry_after="3"),
        HTTPStatusError(503, "busy"),
        OK,
    )
    assert retrying(stub, sleep=waits.append)("u", {}, {}) == OK
    assert waits == [3.0, 20.0] and len(stub.requests) == 3


@pytest.mark.parametrize(
    "error", [HTTPStatusError(400, "bad request"), HTTPStatusError(429, "still")]
)
def test_a_bad_request_or_exhausted_retries_raise(error):
    stub = Stub(*[error] * 4)
    with pytest.raises(HTTPStatusError):
        retrying(stub, sleep=lambda s: None)("u", {}, {})
    assert len(stub.requests) == (1 if error.code == 400 else 4)


def test_an_error_inside_a_200_is_raised_not_parsed():
    client = OpenRouterClient(
        api_key="k",
        transport=Stub({"error": {"message": "provider down", "code": 502}}),
    )
    with pytest.raises(RuntimeError, match="provider down"):
        client.converse("sys", [UserText("hi")], [])


def test_make_client_picks_each_adapter(monkeypatch):
    assert type(make_client("openai")) is OpenAICompatClient
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    client = make_client("openrouter", max_tokens=4096)
    assert type(client) is OpenRouterClient and client.max_tokens == 4096
    with pytest.raises(ValueError, match="unknown llm"):
        make_client("gpt")


def test_make_client_bedrock_needs_a_model_id(monkeypatch):
    monkeypatch.delenv("CIA_BEDROCK_MODEL_ID", raising=False)
    with pytest.raises(
        ValueError, match="CIA_BEDROCK_MODEL_ID"
    ):  # fails before boto3 is imported
        make_client("bedrock")
    assert BedrockClient  # the adapter stays importable without AWS


def test_monitor_lists_openrouter_only_with_a_key(monkeypatch, tmp_path):
    truth = tmp_path / "t.json"
    truth.write_text('{"cases": []}')
    mon = Monitor(str(tmp_path / "x.db"), str(truth), str(tmp_path / "runs"))
    assert not [m for m in mon.models() if m["llm"] == "openrouter"]
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert {"llm": "openrouter", "model": DEFAULT_OPENROUTER_MODEL} in mon.models()
