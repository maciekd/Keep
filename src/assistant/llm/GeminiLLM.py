import base64
import hashlib
import io
import threading
import time
from dataclasses import dataclass
from typing import Generator
from google.genai.types import ThinkingLevel

from .AbstractLLM import AbstractLLM, LLMUnavailable, Thinking
from google import genai
from google.genai import types
from common.logging import trace, info, error
from common.ConfigManager import config

# Gemini rejects requests over 20 MB, so clips above this go through the Files API instead of an
# inline Blob (the margin leaves room for the prompt). The Files API takes files up to 2 GB.
INLINE_VIDEO_LIMIT = 19 * 1024 * 1024
MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024
# An uploaded video is PROCESSING for a few seconds before it can be referenced.
FILE_PROCESSING_TIMEOUT = 120
FILE_POLL_INTERVAL = 1.0

# Explicit context cache of a bulk request's system instruction (the photo indexer's memory bank
# and instructions, 2–7k tokens that are identical from photo to photo and clip to clip). Cached tokens are billed
# at a tenth of the input price; storage is $0.50 per million tokens per hour, so a cache is kept
# only as long as requests keep arriving: every use that finds it in the last half of its life
# extends it by one more TTL, and a trickle lets it lapse. A batch job can take long to start,
# so a cache made for one lives longer and is extended while the job is polled.
SYSTEM_CACHE_TTL_SECONDS = 600
BATCH_CACHE_TTL_SECONDS = 3600
# Inline batch jobs must stay under 20 MB in total; the photo indexer's 1024 px JPEGs are
# ~150 kB each, so a 100-photo fetch fits in one job and a heavier one is split. The limit is
# measured on the wire, where an inline image is base64 — a third larger than its bytes.
INLINE_BATCH_LIMIT = 18 * 1024 * 1024
BASE64_EXPANSION = 4 / 3
BATCH_POLL_SECONDS = 15
_BATCH_ENDED = ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
                "JOB_STATE_PARTIALLY_SUCCEEDED")


@dataclass
class _SystemCache:
    name: str
    tokens: int
    ttl: float
    expires_at: float    # time.monotonic()


# A 429 is Google's spend ceiling ($10 per rolling ten minutes on a fresh account) or a per-minute
# quota: it clears by itself, so background work waits this long, doubling while it keeps coming.
RATE_BACKOFF_SECONDS = 60
RATE_BACKOFF_MAX_SECONDS = 600
# How often a user's own request may try Gemini while the credits are known to be gone.
CREDITS_RECHECK_SECONDS = 60


class _Outage:
    """What Gemini is refusing right now, shared by every GeminiLLM — there is one API key.
    `credits`: a 402, the prepaid balance is gone and every request fails until the user tops up,
    so every call fails fast; a probe (the Photos health check) or, once per
    CREDITS_RECHECK_SECONDS, a user's own request is the live attempt that finds out whether the
    account was topped up — the chat is never locked out until a restart. `rate`: a 429,
    background work backs off until `until` while a user's own request still goes through (it
    would be refused too, but it must find that out itself rather than be told no for a minute)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.kind: str | None = None
        self.until: float | None = None    # time.monotonic(); None = until a probe succeeds
        self.reason: str = ""
        self.backoff = RATE_BACKOFF_SECONDS
        self.recheck_at: float = 0.0    # credits: when the next user request may try again

    def active(self, background: bool) -> str | None:
        with self.lock:
            if self.kind is None:
                return None
            if self.until is not None and time.monotonic() >= self.until:
                return None
            if self.kind == "rate" and not background:
                return None
            if self.kind == "credits" and not background and time.monotonic() >= self.recheck_at:
                self.recheck_at = time.monotonic() + CREDITS_RECHECK_SECONDS
                trace("Gemini: credits outage — letting this user request through to see whether the account was topped up")
                return None
            return self.reason

    def note(self, e: Exception, model: str) -> LLMUnavailable | None:
        text = str(e)
        code = getattr(e, "code", None)    # google.genai.errors.APIError carries the HTTP status
        lowered = text.lower()
        with self.lock:
            if code == 402 or "prepayment credits" in lowered or "credits are depleted" in lowered:
                first = self.kind != "credits"
                self.kind, self.until = "credits", None
                self.recheck_at = time.monotonic() + CREDITS_RECHECK_SECONDS
                self.reason = f"{model}: the Gemini prepaid credits are depleted (402) — every request fails until the account is topped up"
                if first:
                    error(f"Gemini: {self.reason}")
            elif code == 429 or ("429" in text and "RESOURCE_EXHAUSTED" in text):
                if self.kind == "rate" and self.until is not None and time.monotonic() < self.until:
                    return LLMUnavailable(self.reason)
                wait = self.backoff
                self.backoff = min(self.backoff * 2, RATE_BACKOFF_MAX_SECONDS)
                self.kind, self.until = "rate", time.monotonic() + wait
                self.reason = f"{model}: rate limited by Gemini (429) — background requests wait {wait:.0f}s"
                info(f"Gemini: {self.reason} ({text[:120]})")
            else:
                return None
            return LLMUnavailable(self.reason)

    def clear(self):
        with self.lock:
            if self.kind is not None:
                info(f"Gemini: takes requests again (was: {self.reason})")
            self.kind, self.until, self.reason = None, None, ""
            self.backoff = RATE_BACKOFF_SECONDS
            self.recheck_at = 0.0


_outage = _Outage()


class GeminiLLM(AbstractLLM):
    """Talks to Google Gemini through the google-genai client, with native thinking levels."""

    _LOW_FLOOR_GENERATIONS = ('gemini-3.7', 'gemini-3.8')

    def __init__(self, model: str = "gemini-3.5-flash"):
        self.model = model
        self.client = None
        # system-instruction digest -> cache; None marks text Gemini refused to cache (too short)
        self._system_caches: dict[str, _SystemCache | None] = {}
        self._cache_lock = threading.Lock()

    # --- outage (402 / 429) ---------------------------------------------------------------------

    def _request(self, fn, background: bool = False, probing: bool = False):
        """Run one API call: refused outright while Gemini is known to refuse (LLMUnavailable
        without a call — except for the probe, whose job is to find out), a 402 or 429 it
        returns is recorded for every other caller and raised as LLMUnavailable, and a success
        clears whatever was recorded."""
        reason = None if probing else _outage.active(background)
        if reason:
            raise LLMUnavailable(reason)
        try:
            result = fn()
        except Exception as e:
            unavailable = _outage.note(e, self.model)
            if unavailable is not None:
                raise unavailable from e
            raise
        _outage.clear()
        return result

    def _open_stream(self, fn):
        """Start a stream inside the outage bookkeeping. generate_content_stream is lazy — nothing
        reaches Gemini until the first chunk is pulled — so the first `next()` is the call that
        meets a 402 or 429, or proves Gemini answers; `_request` on the bare generator would clear
        an outage before any HTTP call and let the stream's own refusal escape unrecorded."""
        def opened():
            chunks = iter(fn())
            return next(chunks, None), chunks
        first, rest = self._request(opened)
        if first is None:
            trace(f"Gemini: {self.model} answered the stream with no chunk at all")
            return
        yield first
        yield from rest

    def unavailable(self) -> str | None:
        return _outage.active(background=True)

    def out_of_credits(self) -> bool:
        with _outage.lock:
            return _outage.kind == "credits"

    def probe(self) -> bool:
        """One token's worth of generation. True (and the outage cleared) when Gemini answers."""
        self._ensure_initialized()
        try:
            self._request(lambda: self.client.models.generate_content(
                model=self.model, contents="Reply with the word ok.",
                config=types.GenerateContentConfig(max_output_tokens=5)), probing=True)
            return True
        except LLMUnavailable as e:
            trace(f"Gemini probe: still refused — {e}")
            return False
        except Exception as e:
            trace(f"Gemini probe: failed on something else — {e}")
            return False

    @property
    def api_key(self) -> str | None:
        """Read at every use, not captured at construction: a key stored from the chat (config_request)
        must reach the next request without a restart — initialize() builds the client per request."""
        return config.get_api_key('gemini')

    def name(self) -> str:
        return "Gemini"

    def api_key_name(self) -> str:
        return "api_keys.gemini"

    def missing_api_key(self) -> str | None:
        return None if self.api_key else self.api_key_name()

    def prefers_prepended_user_query(self) -> bool:
        return True

    def supports_native_thinking(self) -> bool:
        """Gemini supports native thinking via ThinkingLevel."""
        return True

    def _min_thinking_level(self) -> ThinkingLevel:
        """Lowest thinking level the model accepts.

        Gemini 3.7 and 3.8 Flash reject MINIMAL (400 INVALID_ARGUMENT) — their
        floor is LOW. Older Flash models (3, 3.5, 3.6) still accept MINIMAL.
        """
        if any(gen in self.model for gen in self._LOW_FLOOR_GENERATIONS):
            trace(f"{self.model} does not support MINIMAL thinking — using LOW as floor.")
            return ThinkingLevel.LOW
        return ThinkingLevel.MINIMAL

    def initialize(self):
        """Initialize the Google Generative AI client"""
        self.client = genai.Client(api_key=self.api_key)

    def _convert_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Convert OpenAI/standard message format to Gemini format.

        Handles both string content and list content (Anthropic-format image blocks).

        Returns:
            Tuple of (system_instruction, gemini_messages)
        """
        gemini_messages = []
        system_instruction = None

        for msg in messages:
            if msg['role'] == 'system':
                # The Claude cache markers mean nothing to Gemini — strip them
                # so they don't reach the model as literal text.
                system_instruction = self.strip_cache_markers(msg['content'])
            elif msg['role'] in ('user', 'assistant'):
                role = 'model' if msg['role'] == 'assistant' else 'user'
                content = msg['content']
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if block.get('type') == 'image' and 'source' in block:
                            src = block['source']
                            parts.append(types.Part.from_bytes(
                                data=base64.b64decode(src['data']),
                                mime_type=src['media_type']
                            ))
                        elif block.get('type') == 'text':
                            parts.append({'text': block['text']})
                    gemini_messages.append({'role': role, 'parts': parts})
                else:
                    gemini_messages.append({'role': role, 'parts': [{'text': content}]})

        return system_instruction, gemini_messages

    def create_streaming_completion(self, messages: list[dict], thinking: Thinking | bool = Thinking.NORMAL,
                                    volatile_tail: int = 0, follow_up: bool = False) -> Generator[str, None, None]:
        self._ensure_initialized()

        system_instruction, gemini_messages = self._convert_messages(messages)

        # this is only used for longer queries from the user, we want it to be minimally slower than normal queries. Theres HIGH and MEDIUM here and it's great but doesn't fit what we need there (they're smart but 5s - so too slow)
        thinking_level = ThinkingLevel.LOW if Thinking.of(thinking).full else self._min_thinking_level()
        config_params = {
            'thinking_config': types.ThinkingConfig(thinking_level=thinking_level)
        }

        if system_instruction:
            config_params['system_instruction'] = system_instruction

        generation_config = types.GenerateContentConfig(**config_params) if config_params else None

        chunks = self._open_stream(lambda: self.client.models.generate_content_stream(
            model=self.model,
            contents=gemini_messages,
            config=generation_config
        ))

        for chunk in chunks:
            if chunk.text:
                yield chunk.text

    def create_completion(self, messages: list[dict], thinking: Thinking | bool = Thinking.NORMAL, volatile_tail: int = 0,
                          follow_up: bool = False) -> str:
        self._ensure_initialized()

        system_instruction, gemini_messages = self._convert_messages(messages)

        thinking_level = ThinkingLevel.HIGH if Thinking.of(thinking).full else self._min_thinking_level()
        config_params = {
            'thinking_config': types.ThinkingConfig(thinking_level=thinking_level)
        }

        if system_instruction:
            config_params['system_instruction'] = system_instruction

        generation_config = types.GenerateContentConfig(**config_params) if config_params else None

        response = self._request(lambda: self.client.models.generate_content(
            model=self.model,
            contents=gemini_messages,
            config=generation_config
        ))

        return response.text

    def create_completion_with_images(self, messages: list[dict], images: list, background: bool = False) -> str:
        self._ensure_initialized()

        system_instruction, gemini_messages = self._convert_messages(messages)

        if gemini_messages and gemini_messages[-1]['role'] == 'user':
            parts = gemini_messages[-1]['parts']
            for img in images:
                if isinstance(img, (bytes, bytearray)):
                    mime = "image/png" if img[:4] == b'\x89PNG' else "image/jpeg"
                    parts.insert(-1, types.Part.from_bytes(data=img, mime_type=mime))
                else:
                    parts.insert(-1, types.Part.from_bytes(
                        data=base64.b64decode(img), mime_type="image/jpeg"
                    ))

        # Bulk work (the photo indexer) repeats one long system instruction per image: serve it
        # from an explicit cache. A user's own image question is one-off and goes as is.
        reason = _outage.active(background)
        if reason:
            raise LLMUnavailable(reason)    # before the cache call: no cache for a key that is refused
        cache = self._cached_system(system_instruction, SYSTEM_CACHE_TTL_SECONDS) if background else None
        response = self._generate_with_cache(gemini_messages, system_instruction, cache, background)
        self._log_usage(response, "image")
        return self._text_or_empty(response, "image")

    def _generate_with_cache(self, contents, system_instruction: str | None, cache: _SystemCache | None,
                             background: bool = False):
        """generate_content through `cache` when there is one; a cache Gemini no longer knows
        (expired between the check and the call, deleted elsewhere) is dropped and the request
        repeated with the system instruction inline, so no image is lost to the cache."""
        if cache is not None:
            try:
                return self._request(lambda: self.client.models.generate_content(
                    model=self.model, contents=contents,
                    config=types.GenerateContentConfig(cached_content=cache.name)), background)
            except LLMUnavailable:
                raise
            except Exception as e:
                if not self._is_missing_cache_error(e):
                    raise
                info(f"Gemini: cache {cache.name} is gone ({str(e)[:80]}) — resending with the system instruction inline")
                self._forget_cache(cache)
        config_params = {'system_instruction': system_instruction} if system_instruction else {}
        return self._request(lambda: self.client.models.generate_content(
            model=self.model, contents=contents,
            config=types.GenerateContentConfig(**config_params) if config_params else None), background)

    @staticmethod
    def _is_missing_cache_error(e: Exception) -> bool:
        text = str(e)
        return "cachedContent" in text or "CachedContent" in text or ("403" in text and "cache" in text.lower())

    def _cached_system(self, system_instruction: str | None, ttl: float) -> _SystemCache | None:
        """The explicit cache holding `system_instruction` for this model, created on first use and
        extended on every use that finds it past half its life. None when there is nothing to
        cache, when Gemini refused the text (below its minimum size — remembered, so a short bank
        costs one failed attempt and not one per photo) or when the cache call itself failed:
        the request then goes out with the instruction inline, as before."""
        if not system_instruction:
            return None
        key = hashlib.sha256(f"{self.model}\n{system_instruction}".encode()).hexdigest()
        with self._cache_lock:
            if key in self._system_caches:
                cache = self._system_caches[key]
                if cache is None:
                    return None
                # Good for at least half the life the caller asks for: a batch job (an hour) may
                # pick up a cache the live path made for ten minutes, and then it is lengthened.
                if cache.expires_at - time.monotonic() >= ttl / 2:
                    return cache
                try:
                    self.client.caches.update(name=cache.name, config=types.UpdateCachedContentConfig(ttl=f"{int(ttl)}s"))
                    cache.ttl = ttl
                    cache.expires_at = time.monotonic() + ttl
                    trace(f"Gemini: cache {cache.name} extended by {ttl:.0f}s")
                    return cache
                except Exception as e:
                    info(f"Gemini: could not extend cache {cache.name} ({str(e)[:80]}) — making a new one")
                    del self._system_caches[key]
            try:
                created = self.client.caches.create(model=self.model, config=types.CreateCachedContentConfig(
                    system_instruction=system_instruction, ttl=f"{int(ttl)}s", display_name="floxy-system"))
            except Exception as e:
                if "minimum" in str(e).lower() or "too small" in str(e).lower() or "400" in str(e):
                    info(f"Gemini: system instruction of {len(system_instruction)} chars is below the cache minimum "
                         f"({str(e)[:100]}) — sending it inline from now on")
                    self._system_caches[key] = None
                else:
                    info(f"Gemini: could not create a context cache ({str(e)[:120]}) — sending the system instruction inline")
                return None
            tokens = getattr(getattr(created, 'usage_metadata', None), 'total_token_count', 0) or 0
            cache = _SystemCache(name=created.name, tokens=tokens, ttl=ttl, expires_at=time.monotonic() + ttl)
            self._system_caches[key] = cache
            info(f"Gemini: cached the system instruction as {created.name} ({tokens} tokens, ttl {ttl:.0f}s) "
                 f"— every request on it pays a tenth for those tokens")
            return cache

    def _forget_cache(self, cache: _SystemCache):
        with self._cache_lock:
            for key, known in list(self._system_caches.items()):
                if known is not None and known.name == cache.name:
                    del self._system_caches[key]

    def create_batched_image_completions(self, system_message: str, requests: list[tuple[str, bytes]],
                                         max_wait_seconds: float = 1800) -> list[str | None] | None:
        """The requests as inline batch jobs (one per INLINE_BATCH_LIMIT of images), every token at
        half price, polled until they end or `max_wait_seconds` pass — a job still running then is
        cancelled and its requests reported unanswered. The system message rides in an explicit
        cache when Gemini takes it, so a batched photo pays half of a tenth for the memory bank."""
        self._ensure_initialized()
        if not requests:
            return []
        reason = _outage.active(background=True)
        if reason:
            info(f"Gemini batch: not submitted — {reason}")
            return None
        system_instruction = self.strip_cache_markers(system_message)
        cache = self._cached_system(system_instruction, BATCH_CACHE_TTL_SECONDS)
        request_config = {'cached_content': cache.name} if cache else {'system_instruction': system_instruction}
        jobs: list[tuple[object, list[int]]] = []    # (job, indexes of `requests` it carries)
        t_start = time.time()
        for indexes in self._batch_chunks(requests):
            inline = [{
                'contents': [{'role': 'user', 'parts': [
                    {'inline_data': {'data': requests[i][1], 'mime_type': self._image_mime(requests[i][1])}},
                    {'text': requests[i][0]}]}],
                'config': request_config,
            } for i in indexes]
            size_mb = sum(len(requests[i][1]) for i in indexes) / 1024 / 1024
            try:
                job = self.client.batches.create(model=self.model, src=inline,
                                                 config={'display_name': 'floxy-photo-index'})
            except Exception as e:
                # Tier 1 allows 3M enqueued tokens at a time: a refused job means the caller indexes
                # these photos live, at full price, and the log says why. A 402 or 429 is recorded
                # so the live path does not then try a hundred times.
                _outage.note(e, self.model)
                error(f"Gemini batch: could not submit {len(indexes)} image request(s) ({size_mb:.1f} MB): {str(e)[:160]}")
                self._cancel_jobs(jobs)
                return None
            info(f"Gemini batch {job.name}: {len(indexes)} image request(s) ({size_mb:.1f} MB) submitted, "
                 f"waiting up to {max_wait_seconds / 60:.0f} min")
            jobs.append((job, indexes))

        replies: list[str | None] = [None] * len(requests)
        deadline = t_start + max_wait_seconds
        pending = list(jobs)
        while pending:
            if cache is not None:
                self._cached_system(system_instruction, BATCH_CACHE_TTL_SECONDS)    # keep it alive for the job
            still = []
            for job, indexes in pending:
                try:
                    job = self.client.batches.get(name=job.name)
                except Exception as e:
                    info(f"Gemini batch {job.name}: poll failed ({str(e)[:80]}), looking again later")
                    still.append((job, indexes))
                    continue
                state = self._job_state(job)
                if state not in _BATCH_ENDED:
                    trace(f"Gemini batch {job.name}: {state}")
                    still.append((job, indexes))
                    continue
                answered = self._collect_batch(job, indexes, replies)
                info(f"Gemini batch {job.name}: {state} after {(time.time() - t_start) / 60:.1f} min, "
                     f"{answered}/{len(indexes)} answered")
            pending = still
            if pending and time.time() >= deadline:
                info(f"Gemini batch: {len(pending)} job(s) still running after {max_wait_seconds / 60:.0f} min — "
                     f"cancelling, the caller indexes those photos live")
                self._cancel_jobs(pending)
                break
            if pending:
                time.sleep(BATCH_POLL_SECONDS)
        return replies

    @staticmethod
    def _batch_chunks(requests: list[tuple[str, bytes]]) -> list[list[int]]:
        chunks: list[list[int]] = [[]]
        size = 0
        for i, (_, image) in enumerate(requests):
            encoded = int(len(image) * BASE64_EXPANSION)
            if chunks[-1] and size + encoded > INLINE_BATCH_LIMIT:
                chunks.append([])
                size = 0
            chunks[-1].append(i)
            size += encoded
        return chunks

    @staticmethod
    def _image_mime(image: bytes) -> str:
        return "image/png" if image[:4] == b'\x89PNG' else "image/jpeg"

    @staticmethod
    def _job_state(job) -> str:
        state = getattr(job, 'state', None)
        return getattr(state, 'name', None) or str(state or "")

    def _collect_batch(self, job, indexes: list[int], replies: list[str | None]) -> int:
        """Copy a finished job's replies into `replies` (by request index); returns how many it
        answered. One usage line per job: the per-image accounting the live path logs, summed."""
        responses = list(getattr(getattr(job, 'dest', None), 'inlined_responses', None) or [])
        if len(responses) != len(indexes):
            info(f"Gemini batch {job.name}: {len(responses)} response(s) for {len(indexes)} request(s)")
        answered = 0
        prompt = cached = out = thoughts = 0
        for i, item in zip(indexes, responses):
            response = getattr(item, 'response', None)
            if response is None:
                trace(f"Gemini batch {job.name}: request {i} failed: {str(getattr(item, 'error', ''))[:120]}")
                continue
            usage = getattr(response, 'usage_metadata', None)
            if usage is not None:
                prompt += usage.prompt_token_count or 0
                cached += usage.cached_content_token_count or 0
                out += usage.candidates_token_count or 0
                thoughts += usage.thoughts_token_count or 0
            text = self._text_or_empty(response, "batched image")
            if text:
                replies[i] = text
                answered += 1
        info(f"Gemini batch {job.name}: {answered} reply(ies) used {prompt} prompt tokens ({cached} cached), "
             f"{out} output, {thoughts} thinking — all at batch (half) price")
        return answered

    def _cancel_jobs(self, jobs):
        for job, _ in jobs:
            try:
                self.client.batches.cancel(name=job.name)
                trace(f"Gemini batch {job.name}: cancelled")
            except Exception as e:
                trace(f"Gemini batch {job.name}: could not cancel: {e}")

    @staticmethod
    def _log_usage(response, what: str):
        """One line of token accounting per media request: the prompt tokens (the image or clip
        plus the memory bank and the instructions), how many of them Gemini's implicit cache
        served, and the output split into reply and thinking. The photo backfill is thousands of
        these calls, so this line is where its cost is read off the log."""
        usage = getattr(response, 'usage_metadata', None)
        if usage is None:
            trace(f"Gemini: {what} request returned no usage metadata")
            return
        info(f"Gemini: {what} request used {getattr(usage, 'prompt_token_count', '?')} prompt tokens "
             f"({getattr(usage, 'cached_content_token_count', None) or 0} cached), "
             f"{getattr(usage, 'candidates_token_count', '?')} output, "
             f"{getattr(usage, 'thoughts_token_count', None) or 0} thinking, "
             f"{getattr(usage, 'total_token_count', '?')} total")

    @staticmethod
    def _text_or_empty(response, what: str) -> str:
        """The reply text, or "" when there is no text part at all (safety block, empty
        candidate). Callers treat "" as a parse failure and retry once, so a photo or clip is
        not written off on a single blocked reply — and a None never reaches their parsers."""
        if response.text is not None:
            return response.text
        candidate = next(iter(getattr(response, 'candidates', None) or []), None)
        feedback = getattr(response, 'prompt_feedback', None)
        info(f"Gemini: {what} request returned no text "
             f"(finish_reason={getattr(candidate, 'finish_reason', None)}, "
             f"block_reason={getattr(feedback, 'block_reason', None)})")
        return ""

    def supports_video(self) -> bool:
        return True

    def max_video_bytes(self) -> int:
        return MAX_VIDEO_BYTES

    def think_with_video(self, system_message: str, user_message: str, video_bytes: bytes,
                         mime_type: str, fps: float | None = None) -> str:
        """One-shot video request. Clips up to INLINE_VIDEO_LIMIT are sent inline (bytes + mime
        type); larger ones are uploaded through the Files API first and deleted afterwards. Either
        way the clip precedes the user text. `fps` overrides Gemini's default 1 frame/s sampling;
        None leaves it to the model."""
        return self._retry(lambda: self._create_completion_with_video(
            system_message, user_message, video_bytes, mime_type, fps))

    def _create_completion_with_video(self, system_message: str, user_message: str, video_bytes: bytes,
                                      mime_type: str, fps: float | None) -> str:
        self._ensure_initialized()

        # The photo indexer is the one caller, and its system instruction (the video analyst's
        # instructions and the memory bank) is the same from clip to clip: serve it from the same
        # explicit cache the stills use, so a clip pays full price for its frames only.
        system_instruction = self.strip_cache_markers(system_message)
        reason = _outage.active(background=True)
        if reason:
            # before the upload and the cache call: no file on Gemini, and no cache, for a request that is refused
            raise LLMUnavailable(reason)

        video_metadata = types.VideoMetadata(fps=fps) if fps else None
        size_mb = len(video_bytes) / 1024 / 1024
        uploaded = None
        if len(video_bytes) > INLINE_VIDEO_LIMIT:
            uploaded = self._upload_video(video_bytes, mime_type)
            video_part = types.Part(
                file_data=types.FileData(file_uri=uploaded.uri, mime_type=mime_type),
                video_metadata=video_metadata,
            )
        else:
            video_part = types.Part(
                inline_data=types.Blob(data=video_bytes, mime_type=mime_type),
                video_metadata=video_metadata,
            )
        contents = [{'role': 'user', 'parts': [video_part, {'text': user_message}]}]
        cache = self._cached_system(system_instruction, SYSTEM_CACHE_TTL_SECONDS)

        info(f"Gemini: sending video ({size_mb:.1f} MB, {mime_type}"
             f"{f', {fps} fps' if fps else ''}, {'Files API' if uploaded else 'inline'}"
             f"{f', system instruction from {cache.name}' if cache else ''}) to {self.model}")
        try:
            response = self._generate_with_cache(contents, system_instruction, cache, background=True)
        finally:
            if uploaded is not None:
                self._delete_file(uploaded)

        self._log_usage(response, "video")
        return self._text_or_empty(response, "video")

    def _upload_video(self, video_bytes: bytes, mime_type: str):
        """Upload a clip through the Files API and wait until Gemini has processed it. Returns the
        File; raises when the upload fails, processing fails, or it takes longer than
        FILE_PROCESSING_TIMEOUT. The caller deletes the file when done with it."""
        size_mb = len(video_bytes) / 1024 / 1024
        info(f"Gemini: uploading video ({size_mb:.1f} MB, {mime_type}) through the Files API")
        started = time.monotonic()
        uploaded = self.client.files.upload(
            file=io.BytesIO(video_bytes), config=types.UploadFileConfig(mime_type=mime_type))
        upload_seconds = time.monotonic() - started
        trace(f"Gemini: uploaded {uploaded.name} in {upload_seconds:.1f}s, state {self._file_state(uploaded)}")

        deadline = started + upload_seconds + FILE_PROCESSING_TIMEOUT
        while self._file_state(uploaded) == "PROCESSING":
            if time.monotonic() > deadline:
                self._delete_file(uploaded)
                raise TimeoutError(f"Gemini: {uploaded.name} still processing after {FILE_PROCESSING_TIMEOUT}s")
            time.sleep(FILE_POLL_INTERVAL)
            uploaded = self.client.files.get(name=uploaded.name)

        state = self._file_state(uploaded)
        if state != "ACTIVE":
            self._delete_file(uploaded)
            raise RuntimeError(f"Gemini: {uploaded.name} ended in state {state}")
        info(f"Gemini: {uploaded.name} active after {time.monotonic() - started:.1f}s "
             f"(upload {upload_seconds:.1f}s)")
        return uploaded

    @staticmethod
    def _file_state(file) -> str:
        state = getattr(file, 'state', None)
        return getattr(state, 'name', None) or str(state or "").upper()

    def _delete_file(self, file):
        """Best effort: files expire after 48 h anyway, but the project store is capped at 20 GB."""
        try:
            self.client.files.delete(name=file.name)
            trace(f"Gemini: deleted {file.name}")
        except Exception as e:
            trace(f"Gemini: could not delete {file.name}: {e}")
