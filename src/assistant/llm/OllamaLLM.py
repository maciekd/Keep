import json
import os
import re
import threading
import time
from typing import Generator
from urllib.parse import urlparse

import requests

from .AbstractLLM import AbstractLLM, LLMPreempted, Thinking
from .BackgroundSlot import BackgroundSlot
from .PrefillTracker import LogFormat, PrefillTracker, is_local_host, parse_last
from assistant.commands.formats import QwenFormat as qwen
from assistant.commands.formats.CommandFormats import CommandFormats
from common.logging import info, trace, note, fine
from common.ConfigManager import config
from common.Paths import LOG_DIR


class OllamaLogFormat(LogFormat):
    """The prefill lines of Ollama's debug log (OLLAMA_DEBUG=1), in the two runners' spellings.

    The workload is logged once at the start of a request —
    llama.cpp: cache.go:151 msg="loading cache slot" id=0 cache=6533 prompt=6344 used=5786 remaining=558
    MLX:       prefix_cache.go:124 msg="cache hit" total=25572 matched=25553 cached=24575 left=997
    — and only the MLX engine logs progress after that, every ~2k tokens:
    pipeline.go:200 msg="Prompt processing progress" processed=2048 total=17531
    """
    def work(self, text: str) -> tuple[int, int] | None:
        match = re.search(r'loading cache slot.*?used=(\d+)\s+remaining=(\d+)', text)
        if match is None:
            match = re.search(r'msg="cache (?:hit|miss)".*?cached=(\d+)\s+left=(\d+)', text)
        return (int(match.group(1)), int(match.group(2))) if match else None

    def progress(self, text: str) -> tuple[int, int] | None:
        return parse_last(r'Prompt processing progress.*?processed=(\d+)\s+total=(\d+)', text)


class OllamaLLM(AbstractLLM):
    """Runs a local model through Ollama's native API, with KV-cache pre-heating and prefill diagnostics.

    Cache-slot metrics and prefill progress are read from Ollama's debug log (over ssh when
    base_url is remote) by the class's one `PrefillTracker` — Ollama serves one request at a time,
    and all OllamaLLM tiers are expected to share one server; the tracker's owner token keeps a
    pre-heat that overlaps a user request from touching the request's state.
    """
    _ollama_log_path = str(LOG_DIR / 'ollama.log')
    # The seed rate stands for the first request; each reply's prompt_eval_duration refines it.
    _prefill = PrefillTracker("Ollama", OllamaLogFormat(), seed_tps=350.0)
    _last_payload = ""
    _last_rendered_prompt = ""
    _pre_heat_lock = threading.Lock()
    _pre_heat_in_flight = False
    _pre_heat_pending_messages: list[dict] | None = None
    # (connect, read) timeouts. Real requests deliberately have none — a slow
    # generation must never be killed mid-stream. Pre-heats MUST have one: they
    # hold _pre_heat_in_flight, so a hung socket to a dead remote host would
    # silently disable all future pre-heats. Rendering does no inference and
    # should be quick; a pre-heat prefill of a large prompt can take minutes.
    _RENDER_TIMEOUT = (5, 30)
    _PRE_HEAT_TIMEOUT = (5, 600)

    # --- Background requests (photo indexing) --------------------------------
    # Ollama serves one request at a time (OLLAMA_NUM_PARALLEL=1, and the model fills the
    # machine), so a background image call goes through the single-slot gate (BackgroundSlot):
    # it waits until nothing interactive is in flight and the runner has been idle for
    # `ollama.background_idle_seconds`, and an interactive request (or the pre-heat that
    # follows a turn) arriving mid-image shuts its socket — Ollama drops a sequence whose
    # client is gone — and takes the slot. The evicted caller gets LLMPreempted.
    _slot = BackgroundSlot("Ollama")
    _bg_completed = 0                   # background requests finished since the last snapshot refresh
    _last_pre_heat_messages: list[dict] | None = None   # what the last pre-heat warmed
    # A night of image requests each leaves a prefix-cache snapshot behind; every this many the
    # chat snapshot is re-warmed so the morning's first turn does not pay a cold prefill.
    _SNAPSHOT_REFRESH_EVERY = 50

    # Native thinking is on for a model that has it unless `ollama.think` says otherwise: a
    # reasoning model denied its channel invents one (Qwen wrote its plan inside made-up
    # <system> tags, which reached the user verbatim). The modelfile's samplers are the
    # thinking-mode ones, so no override. The chat template renders every history turn
    # identically with or without thinking — only the generation suffix differs — so the KV
    # prefix is unaffected. `ollama.think` sets the channel: a level — low, medium, high — is
    # passed to Ollama as the `think` value, true leaves the level to the model, false turns the
    # channel off (the prompt's <thinking> instruction then applies, as for any model without
    # one). The default is medium: on the prompt test suite qwen3.8:27b-mlx passed 94.9–95.5%
    # at medium against 92.1–92.7% at true, 93.3–93.8% at low, 93.8% at high and 89.3% off
    # (two runs each, 2026-10-21), at the same wall time — the level steers the model more than
    # it lengthens the reasoning (166 vs 160 output tokens a call). A model that takes no
    # levels needs `ollama.think: true`.
    THINK_LEVELS = ('low', 'medium', 'high')
    DEFAULT_THINK = 'medium'
    # (model, base_url) -> the capabilities /api/show lists; None when the probe failed.
    _capabilities: dict[tuple[str, str], set[str] | None] = {}

    # (model, base_url) pairs already logged — the handler pool constructs dozens of
    # identical instances (25 handlers x 2 tiers), so log each resolved config once.
    _logged_configs: set[tuple[str, str]] = set()

    DEFAULT_MODEL = 'qwen3.8:27b-mlx'  # the model the household's Ollama server serves

    # Ollama picks the tool-call parser from the model and runs it only when the request
    # declares a tool; with none declared a <tool_call> block is swallowed — in neither
    # `content` nor `tool_calls`. So every request to a model with the "tools" capability
    # declares this one, static tool (part of the stable prefix; a model without the
    # capability answers any declared tool with a 400, and has no parser to swallow a block
    # — takes_tools) and the reader writes each parsed `tool_calls` entry back as call text
    # in the active command format (Qwen's under an Ollama smart tier), so the command
    # processor reads what the model wrote in the syntax it parses. The
    # description is against the drift seen in the probe, where the model held the one
    # declared tool for its only function and loaded a toolkit instead of calling the
    # text-declared one.
    STATIC_TOOL = {"type": "function", "function": {
        "name": "load_toolkit",
        "description": "Loads the named toolkits, references of further functions. Only one of your functions: "
                       "the others are declared in the system prompt and in the toolkits you have loaded, and "
                       "every one of them is called in exactly the same format as this one.",
        "parameters": {"type": "object", "required": ["name"],
                       "properties": {"name": {"type": "array", "items": {"type": "string"}}}}}}

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str | None = None):
        self.model = model
        # Explicit base_url wins; otherwise `ollama.base_url` from config.yaml (e.g. the M4
        # Mac Studio), falling back to localhost.
        self.base_url = base_url or config.get('ollama.base_url', 'http://localhost:11434')
        if (model, self.base_url) not in OllamaLLM._logged_configs:
            OllamaLLM._logged_configs.add((model, self.base_url))
            trace(f"OllamaLLM '{model}' using base_url {self.base_url}")
        self.client = True  # satisfies _ensure_initialized()
        OllamaLLM._configure_log_access(self.base_url)

    @classmethod
    def _configure_log_access(cls, base_url: str):
        """Point the prefill tracker at the right log: the local file when Ollama runs on this
        machine, or ssh to the serving host when base_url is remote (the 'loading cache slot'
        lines only exist over there)."""
        host = urlparse(base_url).hostname
        if is_local_host(host):
            cls._prefill.use_log(cls._ollama_log_path)
        else:
            cls._prefill.use_log(config.get('ollama.remote_log_path', os.path.abspath(cls._ollama_log_path)),
                                 ssh_host=config.get('ollama.ssh_host', host))

    def default_command_format(self):
        """Qwen's <tool_call> syntax: what the model family Ollama serves here was trained on."""
        return qwen.QwenFormat

    def prefers_prepended_user_query(self) -> bool:
        return False

    def prefers_byte_prefix_tail(self) -> bool:
        # The MLX runner restores only at snapshots (8k-token multiples and the point just
        # before the previous reply). A tail the next turn re-renders differently sits before
        # that snapshot, so the runner falls back to the previous one and re-prefills the whole
        # tail (the runner log's "cache hit" line: matched= far above cached=).
        return True

    def name(self) -> str:
        return "Ollama"

    def initialize(self):
        pass

    @staticmethod
    def _convert_anthropic_content(messages: list[dict]) -> list[dict]:
        """Flatten Anthropic-style content blocks into Ollama's native shape.

        Ollama's /api/chat only accepts a string `content` plus an optional
        message-level `images` list of raw base64 strings. When content is a
        list of typed blocks (e.g. camera snapshots from the prompt), join the
        text blocks and move image blocks' base64 data into `images`.
        Messages whose content is already a string pass through untouched.
        """
        converted = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                texts, images = [], []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        texts.append(block.get("text", ""))
                    elif block.get("type") == "image" and "source" in block:
                        images.append(block["source"].get("data", ""))  # raw base64, no data: URI
                new_msg = {**msg, "content": "\n".join(texts)}
                if images:
                    new_msg["images"] = images
                converted.append(new_msg)
            elif msg.get("role") == "system":
                # Strip Claude cache markers — deterministic, so Ollama's own
                # KV-cache prefix matching is unaffected.
                converted.append({**msg, "content": OllamaLLM.strip_cache_markers(content)})
            else:
                converted.append(msg)
        return converted

    def _build_request(self, messages: list[dict], stream: bool, num_predict: int | None = None) -> dict:
        req = {
            "model": self.model,
            "messages": self._convert_anthropic_content(messages),
            "stream": stream,
            "keep_alive": -1,
            "think": self.think_setting() if self.thinks() else False,
        }
        if self.takes_tools():
            req["tools"] = [self.STATIC_TOOL]
        if num_predict is not None:
            req["options"] = {"num_predict": num_predict}
        return req

    @classmethod
    def think_setting(cls) -> bool | str:
        """`ollama.think`: one of THINK_LEVELS to pass as the level (DEFAULT_THINK when the key
        is absent), True for native thinking at the model's own level, False for none; another
        spelling is logged and read as the default."""
        value = config.get('ollama.think', cls.DEFAULT_THINK)
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ('true', 'yes', 'on', '1'):
            return True
        if text in ('false', 'no', 'off', '0'):
            return False
        if text in cls.THINK_LEVELS:
            return text
        info(f"ollama.think '{value}' is not true, false or one of {', '.join(cls.THINK_LEVELS)} — using {cls.DEFAULT_THINK}")
        return cls.DEFAULT_THINK

    def thinks(self) -> bool:
        """Whether requests run with native thinking: on unless `ollama.think` is false or
        /api/show says the model has no "thinking" capability (Ollama rejects think=true for
        such a model with a 400)."""
        if self.think_setting() is False:
            return False
        capabilities = self.capabilities()
        return capabilities is None or "thinking" in capabilities

    def takes_tools(self) -> bool:
        """Whether requests declare STATIC_TOOL: on unless /api/show says the model has no
        "tools" capability (Ollama answers a request declaring one with a 400, "does not
        support tools" — gemma3, llava — so such a model gets no tool and its calls, if it
        writes any, come through as text)."""
        capabilities = self.capabilities()
        return capabilities is None or "tools" in capabilities

    def capabilities(self) -> set[str] | None:
        """The capabilities /api/show lists for the model, probed once per (model, base_url);
        None when the host could not be asked, which every capability then assumes it has."""
        key = (self.model, self.base_url)
        if key not in OllamaLLM._capabilities:
            capabilities = None
            try:
                resp = requests.post(f"{self.base_url}/api/show", json={"model": self.model},
                                     timeout=self._RENDER_TIMEOUT)
                resp.raise_for_status()
                capabilities = set(resp.json().get("capabilities", []))
                lacking = [c for c in ("thinking", "tools") if c not in capabilities]
                info(f"Ollama '{self.model}' capabilities: {sorted(capabilities)}"
                     + (f" — running without {' and '.join(lacking)}" if lacking else ""))
            except Exception as e:
                trace(f"Ollama capability probe for '{self.model}' failed ({e}) — assuming it thinks and takes tools")
            OllamaLLM._capabilities[key] = capabilities
        return OllamaLLM._capabilities[key]

    def supports_native_thinking(self) -> bool:
        return self.thinks()

    def handles_thinking_internally(self) -> bool:
        return self.thinks()

    def _rendered_prompt(self, messages: list[dict], num_predict: int | None = None) -> str | None:
        """Ask Ollama to render the final flat prompt without running inference."""
        try:
            req = self._build_request(messages, stream=False, num_predict=num_predict)
            req["_debug_render_only"] = True
            resp = requests.post(f"{self.base_url}/api/chat", json=req, timeout=self._RENDER_TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("_debug_info", {}).get("rendered_template")
        except Exception as e:
            trace(f"Rendered prompt fetch failed (non-critical): {e}")
            return None

    def pre_heat_cache(self, messages: list[dict]):
        """Pre-heat Ollama's KV cache with in-flight guard.
        At most one request is ever active in Ollama. If a pre-heat is already in flight,
        the latest prompt is stored and fired automatically when the current one completes."""
        OllamaLLM._last_pre_heat_messages = messages
        with OllamaLLM._pre_heat_lock:
            if OllamaLLM._pre_heat_in_flight:
                OllamaLLM._pre_heat_pending_messages = messages
                trace("Pre-heat deferred (another request in flight).")
                return
            OllamaLLM._pre_heat_in_flight = True

        self._fire_pre_heat(messages)

    # Start of the chat template's generation suffix appended after the last
    # turn (qwen/ChatML: "<|im_end|>\n<|im_start|>assistant\n<think>...").
    _GENERATION_SUFFIX_MARK = "<|im_end|>\n<|im_start|>assistant"

    def _pre_heat_raw_truncated(self, messages: list[dict], refresh: bool = False) -> bool:
        """Prefill via /api/generate raw=true with the rendered prompt cut at the
        end of the last turn's content. `refresh` is the background snapshot refresh: it
        already holds the background slot and must not restart the idle clock.

        Why not a plain chat request: the MLX runner snapshots the KV cache at
        end-of-prompt minus 4 tokens (pipeline.go, preThinking). A chat pre-heat
        ends with the template's generation suffix (~10 tokens), so that snapshot
        lands PAST the point where the next real request diverges (its trailing
        turn gets the time note and query appended) — unusable, and the runner
        falls back to an older snapshot, re-prefilling the whole tail. Cutting
        the suffix parks the snapshot inside the shared prefix; the next request
        restores from it and prefills only its own tail (measured: 30 tokens
        instead of 1154 on a 1156-token prompt).

        Returns False when the rendered prompt or the suffix marker is
        unavailable — caller falls back to the chat-based pre-heat."""
        rendered = self._rendered_prompt(messages, num_predict=1)
        if rendered is None:
            return False
        idx = rendered.rfind(self._GENERATION_SUFFIX_MARK)
        if idx <= 0:
            trace("Pre-heat: generation suffix marker not found in rendered prompt — "
                  "falling back to chat pre-heat.")
            return False
        truncated = rendered[:idx]
        if not refresh:
            self._interactive_begin()
        token = self._prefill.begin(interactive=False)
        try:
            resp = requests.post(f"{self.base_url}/api/generate", json={
                "model": self.model, "prompt": truncated, "raw": True,
                "stream": False, "keep_alive": -1, "options": {"num_predict": 1},
            }, timeout=self._PRE_HEAT_TIMEOUT)
            resp.raise_for_status()
        finally:
            self._prefill.end(token)
            if not refresh:
                self._interactive_end()
        trace(f"Pre-heat: raw prefill of {len(truncated)} chars "
              f"(cut {len(rendered) - idx}-char template suffix).")
        return True

    def _fire_pre_heat(self, messages: list[dict]):
        """Send a pre-heat request to Ollama in a background thread."""
        if not any(m.get("role") == "user" for m in messages):
            # Ollama's chat renderer rejects a system-only request with a 500
            # ("no user query found in messages"). The stub sits after the shared
            # system prefix, so the KV cache real requests reuse is warmed all the same.
            messages = messages + [{"role": "user", "content": " "}]

        def _do_pre_heat():
            try:
                info(f"Pre-heating Ollama KV cache ({self.model})...")
                if not self._pre_heat_raw_truncated(messages):
                    for _ in self.create_streaming_completion(messages, num_predict=1):
                        pass
                info("Pre-heat complete.")
            except Exception as e:
                trace(f"Pre-heat failed (non-critical): {e}")
            finally:
                self._finish_pre_heat()

        thread = threading.Thread(target=_do_pre_heat, name="OllamaPreHeat", daemon=True)
        thread.start()

    def _finish_pre_heat(self):
        """Release the pre-heat guard, or hand the slot straight to the prompt deferred meanwhile."""
        with OllamaLLM._pre_heat_lock:
            pending = OllamaLLM._pre_heat_pending_messages
            OllamaLLM._pre_heat_pending_messages = None
            if pending is None:
                OllamaLLM._pre_heat_in_flight = False
        if pending is not None:
            info("Pre-heat: processing deferred prompt.")
            self._fire_pre_heat(pending)

    def is_local(self) -> bool:
        return True

    @classmethod
    def background_idle_seconds(cls) -> float:
        return float(config.get('ollama.background_idle_seconds', BackgroundSlot.DEFAULT_IDLE_S))

    @classmethod
    def _interactive_begin(cls):
        cls._slot.interactive_begin()

    @classmethod
    def _interactive_end(cls, resets_idle: bool = True):
        cls._slot.interactive_end(resets_idle)

    def background_wait_seconds(self) -> float:
        return OllamaLLM._slot.wait_seconds()

    def _refresh_snapshot_if_due(self):
        """Every _SNAPSHOT_REFRESH_EVERY background requests, re-warm the last pre-heated chat
        prompt: a fraction of a second while its snapshot survives, a full (background) prefill
        once the image snapshots have pushed it out. Runs inside the background slot."""
        OllamaLLM._bg_completed += 1
        if OllamaLLM._bg_completed % OllamaLLM._SNAPSHOT_REFRESH_EVERY:
            return
        messages = OllamaLLM._last_pre_heat_messages
        if messages is None:
            trace("Ollama: snapshot refresh due but nothing has been pre-heated yet — skipping.")
            return
        with OllamaLLM._pre_heat_lock:
            if OllamaLLM._pre_heat_in_flight or OllamaLLM._slot.abort:
                trace("Ollama: snapshot refresh skipped — a pre-heat is in flight or a user request is arriving.")
                return
            OllamaLLM._pre_heat_in_flight = True
        t0 = time.monotonic()
        try:
            if self._pre_heat_raw_truncated(messages, refresh=True):
                info(f"Ollama: chat snapshot refreshed after {OllamaLLM._bg_completed} background requests "
                     f"in {time.monotonic() - t0:.1f}s")
            else:
                trace("Ollama: snapshot refresh could not render the prompt — skipped.")
        except Exception as e:
            trace(f"Ollama: snapshot refresh failed (non-critical): {e}")
        finally:
            self._finish_pre_heat()

    @classmethod
    def _log_prefix_match(cls, messages: list[dict]) -> str:
        """Compare messages against the last COMPLETED Ollama request.
        Returns the serialized payload — caller must pass it to _mark_payload_completed()
        after Ollama finishes processing."""
        payload = json.dumps(messages, ensure_ascii=False)
        if cls._last_payload:
            shared = 0
            for a, b in zip(cls._last_payload, payload):
                if a != b:
                    break
                shared += 1
            total = len(payload)
            pct = (shared / total * 100) if total else 0
            info(f"Ollama prefix match: {shared}/{total} chars ({pct:.1f}% identical) — "
                 f"first diff at char {shared}")
            if shared < total:
                start = max(0, shared - 40)
                end = min(total, shared + 40)
                old_snippet = cls._last_payload[start:end] if shared < len(cls._last_payload) else ""
                new_snippet = payload[start:end]
                trace(f"  prev: ...{old_snippet}...")
                trace(f"  curr: ...{new_snippet}...")
        else:
            info(f"Ollama prefix match: first request ({len(payload)} chars), no previous to compare")
        return payload

    @classmethod
    def _mark_payload_completed(cls, payload: str):
        cls._last_payload = payload

    @classmethod
    def _mark_request_completed(cls, payload: str, rendered_prompt: str | None):
        cls._last_payload = payload
        if rendered_prompt is not None:
            cls._last_rendered_prompt = rendered_prompt

    def _log_rendered_prompt_match(self, messages: list[dict], num_predict: int | None = None) -> tuple[str | None, tuple[int, int] | None]:
        """Compare Ollama's actual rendered prompt against the last completed request."""
        rendered_prompt = self._rendered_prompt(messages, num_predict=num_predict)
        if rendered_prompt is None:
            return None, None

        if OllamaLLM._last_rendered_prompt:
            shared = 0
            for a, b in zip(OllamaLLM._last_rendered_prompt, rendered_prompt):
                if a != b:
                    break
                shared += 1
            total = len(rendered_prompt)
            pct = (shared / total * 100) if total else 0
            info(f"Ollama rendered prompt match: {shared}/{total} chars ({pct:.1f}% identical) — "
                 f"first diff at char {shared}")
            if shared < total:
                start = max(0, shared - 40)
                end = min(total, shared + 40)
                old_snippet = OllamaLLM._last_rendered_prompt[start:end] if shared < len(OllamaLLM._last_rendered_prompt) else ""
                new_snippet = rendered_prompt[start:end]
                trace(f"  prev rendered: ...{old_snippet}...")
                trace(f"  curr rendered: ...{new_snippet}...")
            return rendered_prompt, (shared, total)
        else:
            info(f"Ollama rendered prompt match: first request ({len(rendered_prompt)} chars), no previous to compare")
            return rendered_prompt, None

    def _log_metrics(self, data: dict, rendered_prefix_stats: tuple[int, int] | None = None,
                     token: int | None = None):
        prompt_tokens = data.get("prompt_eval_count", 0)
        prompt_dur_ns = data.get("prompt_eval_duration", 0)
        eval_count = data.get("eval_count", 0)
        eval_dur_ns = data.get("eval_duration", 0)
        total_dur_ns = data.get("total_duration", 0)

        prompt_dur_s = prompt_dur_ns / 1e9
        eval_dur_s = eval_dur_ns / 1e9
        total_dur_s = total_dur_ns / 1e9
        gen_tps = eval_count / eval_dur_s if eval_dur_s else 0

        cache_metrics = self._prefill.work(token, force=True)
        if cache_metrics:
            used, remaining = cache_metrics
            total = used + remaining
            cache_pct = (used / total * 100) if total else 0
            prompt_tps = remaining / prompt_dur_s if prompt_dur_s else 0
            self._prefill.observe_rate(remaining, prompt_dur_s)
            if rendered_prefix_stats:
                rendered_shared, rendered_total = rendered_prefix_stats
                rendered_pct = (rendered_shared / rendered_total * 100) if rendered_total else 0
                note(f"Ollama cache vs rendered prefix: rendered {rendered_shared}/{rendered_total} chars "
                     f"({rendered_pct:.1f}%) | cached {used}/{total} tokens ({cache_pct:.1f}%)")
            note(f"Ollama metrics: ↑ {prompt_tokens} tokens "
                 f"({cache_pct:.1f}% cached, {remaining} eval, {prompt_tps:.1f} t/s, {prompt_dur_s:.2f}s) | "
                 f"↓ {eval_count} tokens ({gen_tps:.1f} t/s, {eval_dur_s:.2f}s) | "
                 f"{total_dur_s:.2f}s total.")
        else:
            prompt_tps = prompt_tokens / prompt_dur_s if prompt_dur_s else 0
            log = OllamaLLM._prefill.log
            hint = (f"check ssh log access to {log.ssh_host}" if log is not None and log.ssh_host
                    else "start Ollama with local_ollama_start.sh")
            note(f"Ollama metrics: ↑ {prompt_tokens} tokens "
                 f"({prompt_tps:.1f} t/s, {prompt_dur_s:.2f}s) | "
                 f"↓ {eval_count} tokens ({gen_tps:.1f} t/s, {eval_dur_s:.2f}s) | "
                 f"{total_dur_s:.2f}s total. (no cache data — {hint})")

    def create_streaming_completion(self, messages: list[dict], thinking: Thinking | bool = Thinking.NORMAL,
                                    num_predict: int | None = None, volatile_tail: int = 0,
                                    follow_up: bool = False) -> Generator[str, None, None]:
        """Native thinking (the chunk's `thinking` field) is yielded wrapped in <thinking> tags,
        so the command processor swallows and times it exactly like a prompted thinking block.
        `thinking` is ignored: the model thinks whenever it can (see thinks())."""
        is_pre_heat = num_predict is not None
        note(f"Sending {'pre-heat' if is_pre_heat else 'actual'} request to ollama...")
        payload_str = self._log_prefix_match(messages)
        rendered_prompt, rendered_prefix_stats = self._log_rendered_prompt_match(messages, num_predict=num_predict)
        self._interactive_begin()
        token = self._prefill.begin(interactive=not is_pre_heat)
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=self._build_request(messages, stream=True, num_predict=num_predict),
                stream=True,
                # The chat-fallback pre-heat must not wedge _pre_heat_in_flight on a
                # dead host; real requests keep no timeout (see _PRE_HEAT_TIMEOUT).
                timeout=self._PRE_HEAT_TIMEOUT if is_pre_heat else None,
            )
            resp.raise_for_status()

            in_thinking = False
            reply = ""  # everything yielded as reply text so far: the line break before a call needs it
            content_seen = ""  # the model's own content alone, for the unparsed-<tool_call> check
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                message = data.get("message", {})
                thought = message.get("thinking", "")
                if thought:
                    self._prefill.first_token(token)
                    if not in_thinking:
                        in_thinking = True
                        yield "<thinking>"
                    yield thought
                content = message.get("content", "")
                if content:
                    self._prefill.first_token(token)
                    if in_thinking:
                        in_thinking = False
                        yield "</thinking>"
                    reply += content
                    content_seen += content
                    yield content
                for call in message.get("tool_calls") or []:
                    text = self._render_tool_call(call)
                    if not text:
                        continue
                    self._prefill.first_token(token)
                    if in_thinking:
                        in_thinking = False
                        yield "</thinking>"
                    if reply and not reply.endswith("\n"):
                        text = "\n" + text
                    reply += text
                    yield text
                if data.get("done"):
                    if in_thinking:
                        trace("Ollama stream ended inside the thinking block — closing it")
                        yield "</thinking>"
                    self._check_unparsed_calls(content_seen)
                    self._log_metrics(data, rendered_prefix_stats=rendered_prefix_stats, token=token)
                    OllamaLLM._mark_request_completed(payload_str, rendered_prompt)
        finally:
            self._prefill.end(token)
            self._interactive_end()

    @classmethod
    def _content_of(cls, data: dict) -> str:
        """The reply text of a non-streaming response — the content, then every parsed tool
        call written back as Qwen text; its native thinking is logged."""
        message = data.get("message", {})
        if message.get("thinking"):
            fine(f"Ollama thought:\n{message['thinking'].strip()}")
        content = message.get("content", "")
        cls._check_unparsed_calls(content)
        for call in message.get("tool_calls") or []:
            if text := cls._render_tool_call(call):
                content += ("\n" if content and not content.endswith("\n") else "") + text
        return content

    @staticmethod
    def _render_tool_call(call: dict) -> str:
        """A `message.tool_calls` entry written back as call text in the *active* command
        format, for the command parser; '' for an entry without a function name. The model
        wrote a <tool_call> block, but the parser reads the smart tier's format — Keep's or
        DSML under a Claude or ds4 smart tier — so the call is rendered in that one. The
        arguments arrive as strings (or, when Ollama decoded them, numbers, booleans and
        lists) and are written back the way the model spelled them."""
        function = call.get("function") or {}
        name = function.get("name")
        if not name:
            trace(f"Ollama tool call without a function name, dropped: {call!r}")
            return ""
        fmt = CommandFormats.active()
        arguments = fmt.arguments_dict(function.get("arguments") or {})
        trace(f"Ollama parsed a {name} call ({', '.join(arguments)}) — written back as {fmt.name} call text")
        return fmt.render_tool_call(name, arguments)

    @staticmethod
    def _check_unparsed_calls(reply: str):
        """Ollama parses every well-formed <tool_call> block out of the content, so one left in
        it is a block the model got wrong; Keep's own parser still reads what it can."""
        if qwen.OPEN in reply:
            note(f"Ollama left <tool_call> text in the content — the model wrote a block its parser "
                 f"did not take; the command parser reads it as written ({reply.count(qwen.OPEN)} opener(s))")

    def create_completion(self, messages: list[dict], thinking: Thinking | bool = Thinking.NORMAL, volatile_tail: int = 0,
                          follow_up: bool = False) -> str:
        payload_str = self._log_prefix_match(messages)
        rendered_prompt, rendered_prefix_stats = self._log_rendered_prompt_match(messages)
        self._interactive_begin()
        token = self._prefill.begin()
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=self._build_request(messages, stream=False),
            )
            resp.raise_for_status()

            data = resp.json()
            self._log_metrics(data, rendered_prefix_stats=rendered_prefix_stats, token=token)
            OllamaLLM._mark_request_completed(payload_str, rendered_prompt)
            return self._content_of(data)
        finally:
            self._prefill.end(token)
            self._interactive_end()

    def create_completion_with_images(self, messages: list[dict], images: list, background: bool = False) -> str:
        import base64

        processed = []
        for msg in messages:
            if msg['role'] == 'user' and images:
                img_b64_list = []
                for image in images:
                    if isinstance(image, (bytes, bytearray)):
                        img_b64_list.append(base64.b64encode(image).decode('utf-8'))
                    elif isinstance(image, str):
                        try:
                            base64.b64decode(image)
                            img_b64_list.append(image)
                        except Exception:
                            with open(image, 'rb') as f:
                                img_b64_list.append(base64.b64encode(f.read()).decode('utf-8'))

                processed.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "images": img_b64_list
                })
                images = []
            else:
                processed.append(msg)

        if background:
            return self._background_completion(processed)

        payload_str = self._log_prefix_match(processed)
        rendered_prompt, rendered_prefix_stats = self._log_rendered_prompt_match(processed)
        self._interactive_begin()
        token = self._prefill.begin()
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=self._build_request(processed, stream=False),
            )
            resp.raise_for_status()

            data = resp.json()
            self._log_metrics(data, rendered_prefix_stats=rendered_prefix_stats, token=token)
            OllamaLLM._mark_request_completed(payload_str, rendered_prompt)
            return self._content_of(data)
        finally:
            self._prefill.end(token)
            self._interactive_end()

    def _background_completion(self, messages: list[dict]) -> str:
        """The abortable image request: waits for the idle slot, streams the answer over a
        connection the gate can shut, and raises LLMPreempted when a user request evicts it.
        No prefix-match diagnostics — an image payload matches nothing, and the last chat
        payload must stay what the next chat turn compares against."""
        body = json.dumps(self._build_request(messages, stream=True))
        slot = OllamaLLM._slot
        slot.acquire(self.background_idle_seconds())
        token = self._prefill.begin(interactive=False)
        chunks: list[str] = []
        try:
            conn = slot.open_connection(self.base_url)
            slot.register_connection(conn)
            conn.request("POST", "/api/chat", body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            if resp.status != 200:
                raise RuntimeError(f"Ollama /api/chat returned {resp.status}: {resp.read(500)!r}")
            for raw in iter(resp.readline, b""):
                line = raw.strip()
                if not line:
                    continue
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                if content:
                    self._prefill.first_token(token)
                    chunks.append(content)
                if data.get("done"):
                    self._log_metrics(data, token=token)
                    break
            else:
                raise RuntimeError("Ollama closed the stream before the response was done")
            conn.close()
            self._refresh_snapshot_if_due()
        except LLMPreempted:
            raise
        except Exception:
            if preempted := slot.preempted(len(chunks)):
                raise preempted from None
            raise
        finally:
            self._prefill.end(token)
            slot.release()
        return "".join(chunks)
