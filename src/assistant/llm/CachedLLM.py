import random
from collections import Counter
from typing import Generator, Optional
from .AbstractLLM import AbstractLLM, Thinking

from common.ConfigManager import config
from common.logging import note, trace, info
from common.FloxyDatabase import FloxyDatabase
from assistant.commands.formats.CommandFormats import CommandFormats


class CachedLLM(AbstractLLM):
    """Wraps another LLM and replays a stored majority response for previously seen user queries.

    Keyed on the clean utterance in cache_query (never on the volatile last message); a hit is
    only served when the cached responses agree on a command set, and the served hit can be
    invalidated afterwards via invalidate_served_response().
    """

    def __init__(self, llm: AbstractLLM):
        self.llm = llm
        self.db = FloxyDatabase.get_instance()
        # Knows no command until update_command_list brings the real list
        self.format = CommandFormats.active()([])
        self.last_response_was_cached = False
        # The clean user utterance to key the cache on (e.g. "turn off the kitchen lights").
        # Set by the caller before each request; None means "do not cache this request".
        self.cache_query: str | None = None
        # Serve-time state for invalidate_served_response: the query and the full set of
        # response wordings sharing the served majority command set. Only populated while
        # the most recent stream was a cache hit.
        self.last_served_query: str | None = None
        self.last_served_responses: list[str] | None = None

    def create_streaming_completion(self, messages: list[dict], thinking: Thinking | bool = Thinking.NORMAL,
                                    volatile_tail: int = 0, follow_up: bool = False) -> Generator[str, None, None]:
        # Key the cache on the clean user utterance supplied via cache_query, NOT on
        # messages[-1]["content"]. The final message carries per-request volatile context
        # (sensor states, temperatures, timestamps, trailing language/brevity directives)
        # that changes every request, so keying on it never produced a hit — the cache had
        # been silently dead (logging "Cachable rows: 0") ever since that context was added.
        self.last_served_query = None
        self.last_served_responses = None
        query = self.cache_query.strip() if isinstance(self.cache_query, str) else ""
        if not query:
            self.last_response_was_cached = False
            yield from self.llm.create_streaming_completion(messages, thinking=thinking, volatile_tail=volatile_tail,
                                                            follow_up=follow_up)
            return
        response_counts = self.db.get_response_counts_for_query(query)
        cachable_count = sum(count for _, count in response_counts)
        trace(f"{query}   Cachable rows: {cachable_count}, unique responses: {len(response_counts)}")
        # Gate on observation count, not unique wordings: 3 identical confirmations are 3 votes
        # that this mapping is stable. The majority vote is row-weighted for the same reason.
        if cachable_count >= 3:
            majority_commands = self.find_majority_commands(response_counts)
            if majority_commands is not None:
                responses = [r for r, _ in response_counts]
                matching = self.filter_responses_by_commands(responses, majority_commands)
                r = random.choice(matching)
                note("LLM response found in cache, returning randomized cached response.")
                trace(f"Using response: {r}")
                self.last_response_was_cached = True
                self.last_served_query = query
                self.last_served_responses = matching
                yield r
                return

        self.last_response_was_cached = False
        yield from self.llm.create_streaming_completion(messages, thinking=thinking)

    def invalidate_served_response(self, reason: str) -> int:
        """Mark every row backing the most recently served cached response as not_cachable.
        Marks all wordings sharing the served majority command set, not just the randomly
        chosen one — they all carry the same bad command mapping."""
        if not self.last_served_query or not self.last_served_responses:
            info("No served cached response recorded — nothing to invalidate.")
            return 0
        marked = self.db.mark_responses_not_cachable(self.last_served_query, self.last_served_responses)
        info(f"Marked {marked} cached row(s) not_cachable for query '{self.last_served_query}' — {reason}.")
        self.last_served_query = None
        self.last_served_responses = None
        return marked

    def create_completion(self, messages: list[dict], thinking: Thinking | bool = Thinking.NORMAL, volatile_tail: int = 0,
                          follow_up: bool = False) -> str:
        self.last_response_was_cached = False
        return self.llm.create_completion(messages, thinking=thinking, volatile_tail=volatile_tail, follow_up=follow_up)

    def create_completion_with_images(self, messages: list[dict], images: list, background: bool = False) -> str:
        self.last_response_was_cached = False
        return self.llm.create_completion_with_images(messages, images, background=background)

    def is_local(self) -> bool:
        return self.llm.is_local()

    def supports_video(self) -> bool:
        return self.llm.supports_video()

    def max_video_bytes(self) -> int:
        return self.llm.max_video_bytes()

    def think_with_video(self, system_message: str, user_message: str, video_bytes: bytes,
                         mime_type: str, fps: float | None = None) -> str:
        self.last_response_was_cached = False
        return self.llm.think_with_video(system_message, user_message, video_bytes, mime_type, fps=fps)

    def initialize(self):
        return self.llm.initialize()

    def api_key_name(self) -> str | None:
        return self.llm.api_key_name()

    def missing_api_key(self) -> str | None:
        return self.llm.missing_api_key()

    @property
    def model(self) -> str:
        return self.llm.model

    def name(self) -> str:
        return self.llm.name()

    def provider(self) -> str:
        return self.llm.provider()

    def display_model(self) -> str:
        return self.llm.display_model()

    def refusal_fallback(self) -> AbstractLLM | None:
        """The inner LLM's fallback, unwrapped: it answers one refused request, which
        is nothing to serve from or record into the response cache."""
        return self.llm.refusal_fallback()

    def prefers_prepended_user_query(self) -> bool:
        return self.llm.prefers_prepended_user_query()

    def set_prompt_cache(self, enabled: bool):
        self.llm.set_prompt_cache(enabled)

    def set_per_request_prefix(self, per_request: bool):
        self.llm.set_per_request_prefix(per_request)

    def supports_native_thinking(self) -> bool:
        return self.llm.supports_native_thinking()

    def handles_thinking_internally(self) -> bool:
        return self.llm.handles_thinking_internally()

    @property
    def last_stop_reason(self) -> str | None:
        """A served cached reply ended nowhere: None, so the wrapped LLM's last stop
        reason is never taken for this reply's."""
        return None if self.last_response_was_cached else self.llm.last_stop_reason

    def output_cap(self) -> int | None:
        return self.llm.output_cap()

    def cancel(self):
        self.llm.cancel()

    def prefers_byte_prefix_tail(self) -> bool:
        return self.llm.prefers_byte_prefix_tail()

    def prefers_query_as_own_turn(self) -> bool:
        return self.llm.prefers_query_as_own_turn()

    def pre_heat_cache(self, messages: list[dict]):
        self.llm.pre_heat_cache(messages)

    def _consolidated_command_set(self, response: str) -> frozenset[tuple[str, str]]:
        return frozenset(self.consolidate_commands(self.extract_all_commands(response)))

    def find_majority_commands(self, response_counts: list[tuple[str, int]]) -> frozenset[tuple[str, str]] | None:
        """Find the command set produced by >=50% of observed rows. Votes are row-weighted:
        a response seen in 2 rows casts 2 votes, so a single stray wording cannot tie a true
        majority. Returns None if no majority is found or the top two sets are tied — a tie
        means Counter ordering would pick the winner arbitrarily, and caching the wrong
        command is worse than not caching."""
        if not response_counts:
            return None

        command_sets = [(self._consolidated_command_set(r), count) for r, count in response_counts]

        # Never cache responses without commands
        if any(len(cs) == 0 for cs, _ in command_sets):
            trace("No commands in one response - we never cache responses without commands.")
            return None

        counts = Counter()
        for cs, count in command_sets:
            counts[cs] += count
        total = sum(counts.values())
        ranked = counts.most_common(2)
        most_common_set, top_count = ranked[0]
        ratio = top_count / total
        trace(f"Most common command set appears in {top_count}/{total} ({ratio:.0%}) of rows.")

        if len(ranked) > 1 and ranked[1][1] == top_count:
            trace("Top command sets are tied - refusing to cache an arbitrary winner.")
            return None

        if ratio >= 0.50:
            return most_common_set

        trace(f"No majority command set found (need 50%, best was {ratio:.0%}).")
        return None

    def filter_responses_by_commands(self, responses: list[str],
                                     target_commands: frozenset[tuple[str, str]]) -> list[str]:
        """Return only responses whose consolidated commands match the target set."""
        return [r for r in responses if self._consolidated_command_set(r) == target_commands]

    @property
    def parser(self):
        """The format under its old name, for a caller that still asks for the parser."""
        return self.format

    def extract_all_commands(self, response: str) -> list[tuple[str, str]]:
        """Extract all commands; params are joined back into a single string so the
        cache layer can hash (cmd, params_str) tuples for equality-based matching.
        The format's structured form (list[str]) is collapsed here because cache
        keys only need string equality, not parameter-boundary preservation."""
        return [(name, ", ".join(params)) for name, params in self.format.extract_all(response)]

    def update_command_list(self, commands_list: list[str]):
        """Rebuild the format's parser with the actual list of available commands."""
        self.format = CommandFormats.active()(commands_list)

    def command_format(self):
        return self.llm.command_format()

    def use_command_format(self, fmt):
        self.llm.use_command_format(fmt)

    @staticmethod
    def consolidate_commands(commands: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """
        Transform commands to optimize and simplify them so that the cache will catch everything
        1. Convert dim(light, 0) to turn_off(light)
        2. Consolidate the device groups of `home.command_consolidation` into their scene
        """
        if len(commands) == 0:
            return []

        refined = []

        # Step 1: Convert dim to 0 -> turn_off
        for cmd, params in commands:
            if cmd == "dim":
                # Parse params string to check if dimming to 0
                param_parts = [p.strip() for p in params.split(',')]
                if len(param_parts) >= 2 and param_parts[-1] == "0":
                    # Convert [dim, Kitchen Lamp, 0] to [turn_off, Kitchen Lamp]
                    light_name = ', '.join(param_parts[:-1])
                    refined.append(("turn_off", light_name))
                else:
                    refined.append((cmd, params))
            else:
                refined.append((cmd, params))

        # Step 2: a group of devices switched together becomes the scene that does the same
        for group in config.get('home.command_consolidation', []) or []:
            refined = CachedLLM._consolidate_group(refined, set(group.get('devices') or []),
                                                   group.get('on_scene'), group.get('off_scene'))
        return refined

    @staticmethod
    def _consolidate_group(commands: list[tuple[str, str]], devices: set[str],
                           on_scene: Optional[str], off_scene: Optional[str]) -> list[tuple[str, str]]:
        """Swap turn_on / turn_off of every device in `devices` for one execute(scene)."""
        if not devices:
            return commands
        for verb, scene in (("turn_on", on_scene), ("turn_off", off_scene)):
            switched = {params for cmd, params in commands if cmd == verb and params in devices}
            if scene and switched == devices:
                trace(f"Consolidating {len(devices)} {verb} commands into execute({scene})")
                commands = [(cmd, params) for cmd, params in commands
                            if not (cmd == verb and params in devices)] + [("execute", scene)]
        return commands
