import collections
from typing import Optional


_EMPTY_DEQUE: collections.deque = collections.deque()


class TTFTMonitor:
    """Keeps recent time-to-first-token and prompt-cache-hit history per LLM model for the DashboardTab.

    `samples` is the one chronological list of every recorded TTFT with the model it came from,
    for the chart that shows the smart and the fast model side by side; `smart_ttft` / `fast_ttft`
    are the per-model views of the same recordings. `cache_samples` is the same for the cache hit
    of every request whose backend reports one — Claude, ds4 and Ollama alike — so a chart can
    tell a hosted model's cache from a local one's."""

    _instance: Optional['TTFTMonitor'] = None
    HISTORY_SIZE = 60

    @classmethod
    def get_instance(cls) -> 'TTFTMonitor':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    def __init__(self):
        self._data: dict[str, collections.deque] = {}
        self._samples: collections.deque = collections.deque(maxlen=self.HISTORY_SIZE)
        self._cache_samples: collections.deque = collections.deque(maxlen=self.HISTORY_SIZE)
        self.smart_model: str = ""
        self.fast_model: str = ""

    def register_models(self, smart_model: str, fast_model: str):
        self.smart_model = smart_model
        self.fast_model = fast_model

    def record(self, model: str, ttft: float):
        if model not in self._data:
            self._data[model] = collections.deque(maxlen=self.HISTORY_SIZE)
        self._data[model].append(ttft)
        self._samples.append((model, ttft))

    def record_cache_pct(self, model: str, pct: int):
        self._cache_samples.append((model, pct))

    @property
    def samples(self) -> collections.deque:
        """(model, ttft) pairs in the order they were recorded, every model together."""
        return self._samples

    @property
    def cache_samples(self) -> collections.deque:
        """(model, cache hit %) pairs in the order the requests came, every model together."""
        return self._cache_samples

    @property
    def smart_ttft(self) -> collections.deque:
        return self._data.get(self.smart_model, _EMPTY_DEQUE)

    @property
    def fast_ttft(self) -> collections.deque:
        return self._data.get(self.fast_model, _EMPTY_DEQUE)

    @staticmethod
    def shorten(model: str) -> str:
        name = model.lower().replace("claude-", "")
        parts = [p for p in name.split("-") if not (len(p) == 8 and p.isdigit())]
        return "-".join(parts)
