from dataclasses import dataclass


@dataclass
class LLMUsage:
    """What one completion used, in the provider's own token counts, and what it cost when the
    backend knows its model's price.

    Re-sent requests of one completion (an empty reply, a retry) are summed into it, so it is
    what the call was billed. `cost_usd` is None when the backend has no price for the model.
    """
    input_tokens: int = 0       # read at the full input price — neither written to nor read from the cache
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    cache_read: int = 0
    output_tokens: int = 0      # thinking included
    batch: bool = False
    cost_usd: float | None = None

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        cost = None if self.cost_usd is None or other.cost_usd is None else self.cost_usd + other.cost_usd
        return LLMUsage(self.input_tokens + other.input_tokens, self.cache_write_5m + other.cache_write_5m,
                        self.cache_write_1h + other.cache_write_1h, self.cache_read + other.cache_read,
                        self.output_tokens + other.output_tokens, self.batch and other.batch, cost)

    def describe(self) -> str:
        cache_written = self.cache_write_5m + self.cache_write_1h
        cost = "cost unknown (no price for the model)" if self.cost_usd is None else f"${self.cost_usd:.2f}"
        return (f"{self.input_tokens} uncached + {cache_written} cache-written + {self.cache_read} cache-read "
                f"input tokens, {self.output_tokens} output tokens, {cost}"
                + (" at batch (half) price" if self.batch else ""))

    @staticmethod
    def of(llm) -> "LLMUsage | None":
        """The usage of `llm`'s most recent completion, None when the backend reports none."""
        usage = getattr(llm, 'last_usage', None)
        return usage if isinstance(usage, LLMUsage) else None

    @staticmethod
    def describe_call(llm) -> str:
        """Why `llm`'s most recent completion ended and what it used, for a log line."""
        usage = LLMUsage.of(llm)
        return (f"stop_reason={getattr(llm, 'last_stop_reason', None) or 'not reported'}, "
                + (usage.describe() if usage else "no usage reported"))
