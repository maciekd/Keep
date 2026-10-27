from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from integrations.podcasts.PodcastsConfig import LlmSettings
from integrations.podcasts.ReportLLM import ReportLLM, ReportLLMError


def response(content=None, calls=()):
    message = SimpleNamespace(content=content, tool_calls=[
        SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name="web_search", arguments=args))
        for i, args in enumerate(calls)] or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def llm(provider="openai", model="gpt-4o", *responses, search=None):
    client = Mock()
    client.chat.completions.create.side_effect = list(responses)
    return ReportLLM(LlmSettings(provider, model, None, "sk-test"), client=client,
                     search=search or (lambda q: [{"title": "T", "url": "https://example.com", "snippet": "s"}])), client


def test_a_search_call_is_answered_and_the_final_text_returned():
    queries = []
    model, client = llm("openai", "gpt-4o", response(calls=['{"query": "Alderbank tokamak"}']), response("Final."),
                        search=lambda q: queries.append(q) or [])
    result = model.complete("system", "user", with_search=True)
    assert result.text == "Final." and result.queries == ["Alderbank tokamak"] == queries
    tool_message = client.chat.completions.create.call_args_list[1].kwargs["messages"][-1]
    assert tool_message["role"] == "tool" and "Alderbank tokamak" in tool_message["content"]
    assert result.prompt_tokens == 20


def test_after_the_search_budget_the_model_must_answer_without_tools():
    model, client = llm("openai", "gpt-4o", response(calls=['{"query": "a"}']), response("Done."))
    model.complete("s", "u", with_search=True, max_tool_rounds=1)
    last = client.chat.completions.create.call_args_list[-1].kwargs
    assert "tools" not in last and "tool-call budget" in last["messages"][-1]["content"]


def test_a_reasoning_model_gets_no_temperature_and_a_developer_message():
    model, client = llm("openai", "o3-mini", response("Ok."))
    model.complete("s", "u")
    request = client.chat.completions.create.call_args.kwargs
    assert "temperature" not in request and request["messages"][0]["role"] == "developer"
    assert "max_completion_tokens" in request


def test_the_local_server_gets_max_tokens():
    model, client = llm("local", "qwen-local", response("Ok."))
    model.complete("s", "u")
    request = client.chat.completions.create.call_args.kwargs
    assert "max_tokens" in request and request["temperature"] == 0.2


def test_an_api_error_and_an_empty_answer_are_report_errors():
    model, _ = llm("openai", "gpt-4o", RuntimeError("quota"))
    with pytest.raises(ReportLLMError, match="quota"):
        model.complete("s", "u")
    model, _ = llm("openai", "gpt-4o", response(""))
    with pytest.raises(ReportLLMError):
        model.complete("s", "u")


def test_no_api_key_is_a_report_error_before_any_request():
    with pytest.raises(ReportLLMError, match="API key"):
        ReportLLM(LlmSettings("openai", "gpt-4o", None, None)).complete("s", "u")
