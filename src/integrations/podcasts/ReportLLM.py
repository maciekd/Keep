"""The chat-completions client the podcast reports are written with: the OpenAI API or a local omlx server.

Both speak the OpenAI chat-completions protocol, so one `openai` client serves both — the local
path only sets `base_url` (omlx on port 8000 by default). `complete` runs one conversation with
an optional `web_search` tool (DuckDuckGo): the model may search up to `max_tool_rounds` times,
then is told to answer with what it has. The OpenAI reasoning models (o1, o3, o4 …) take
neither a temperature nor a system role under that name, so those two are adapted for them.
"""
import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from common.logging import error, info, trace
from integrations.podcasts import web_search
from integrations.podcasts.PodcastsConfig import PROVIDER_LOCAL, LlmSettings

REQUEST_TIMEOUT_S = 20 * 60
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": ("Search the public web and return top results (title, url, snippet). Use this to verify a "
                        "name, paper/book title, company, date, or figure you are not fully certain about before "
                        "stating it as fact in your answer."),
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query."}},
                       "required": ["query"]},
    },
}
OUT_OF_SEARCHES = ("You've used your tool-call budget. Do not call any more tools. Give your complete final Markdown "
                   "answer now, using everything you've learned so far and marking anything you couldn't verify as "
                   "unverified.")
REASONING_MODEL_RE = re.compile(r"^o\d")


class ReportLLMError(Exception):
    """The LLM could not be reached or answered with an error."""


@dataclass
class Completion:
    text: str
    seconds: float
    queries: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ReportLLM:
    """One configured model, called with a system prompt and a user message."""

    def __init__(self, settings: LlmSettings, client=None, search=web_search.search):
        self.settings = settings
        self._client = client
        self._search = search

    @property
    def label(self) -> str:
        return self.settings.label

    def client(self):
        if self._client is None:
            if not self.settings.api_key:
                raise ReportLLMError("no OpenAI API key — set podcasts.openai_api_key or OPENAI_API_KEY")
            from openai import OpenAI
            self._client = OpenAI(api_key=self.settings.api_key, base_url=self.settings.base_url,
                                  timeout=REQUEST_TIMEOUT_S, max_retries=2)
        return self._client

    def _reasoning(self) -> bool:
        return self.settings.provider != PROVIDER_LOCAL and bool(REASONING_MODEL_RE.match(self.settings.model))

    def _request(self, messages: list[dict], tools: Optional[list], max_tokens: int, temperature: float) -> dict:
        request = {"model": self.settings.model, "messages": messages}
        if self.settings.provider == PROVIDER_LOCAL:
            request["max_tokens"] = max_tokens
        else:
            request["max_completion_tokens"] = max_tokens
        if not self._reasoning():
            request["temperature"] = temperature
        if tools:
            request["tools"] = tools
        return request

    def complete(self, system: str, user: str, *, with_search: bool = False, max_tool_rounds: int = 4,
                 max_tokens: int = 12000, temperature: float = 0.2) -> Completion:
        role = "developer" if self._reasoning() else "system"
        messages = [{"role": role, "content": system}, {"role": "user", "content": user}]
        result = Completion(text="", seconds=0.0)
        started = time.time()
        rounds = max_tool_rounds if with_search else 0
        for turn in range(rounds + 1):
            final = turn == rounds
            if final and with_search and turn > 0:
                messages.append({"role": "user", "content": OUT_OF_SEARCHES})
            try:
                response = self.client().chat.completions.create(
                    **self._request(messages, None if final else [WEB_SEARCH_TOOL], max_tokens, temperature))
            except ReportLLMError:
                raise
            except Exception as e:
                error(f"Podcasts: {self.label} request failed: {type(e).__name__}: {e}")
                raise ReportLLMError(f"{self.label}: {type(e).__name__}: {e}") from e
            if response.usage:
                result.prompt_tokens += response.usage.prompt_tokens or 0
                result.completion_tokens += response.usage.completion_tokens or 0
            message = response.choices[0].message
            calls = list(message.tool_calls or [])
            if final or not calls:
                result.text = (message.content or "").strip()
                break
            messages.append({"role": "assistant", "content": message.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                            for c in calls]})
            for call in calls:
                query = self._query_of(call.function.arguments)
                result.queries.append(query)
                try:
                    found = self._search(query)
                except Exception as e:
                    trace(f"Podcasts: web_search({query!r}) raised {type(e).__name__}: {e}")
                    found = []
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": web_search.format_results(query, found)})
        result.seconds = round(time.time() - started, 1)
        if not result.text:
            raise ReportLLMError(f"{self.label} answered with no text")
        info(f"Podcasts: {self.label} answered {len(result.text)} chars in {result.seconds:.0f}s "
             f"({len(result.queries)} searches, {result.prompt_tokens}+{result.completion_tokens} tokens)")
        return result

    @staticmethod
    def _query_of(arguments: str) -> str:
        try:
            parsed = json.loads(arguments or "{}")
            return str(parsed.get("query") or arguments) if isinstance(parsed, dict) else str(arguments)
        except (json.JSONDecodeError, TypeError):
            return str(arguments)
