import pytest
from common.TTFTMonitor import TTFTMonitor


@pytest.fixture(autouse=True)
def reset_singleton():
    TTFTMonitor.reset_instance()
    yield
    TTFTMonitor.reset_instance()


class TestRecordAndRetrieve:
    def test_record_stores_values(self):
        mon = TTFTMonitor()
        mon.register_models("claude-3-opus-20240229", "claude-3-haiku-20240307")
        mon.record("claude-3-opus-20240229", 1.5)
        mon.record("claude-3-opus-20240229", 2.0)
        assert list(mon.smart_ttft) == [1.5, 2.0]

    def test_history_capped_at_history_size(self):
        mon = TTFTMonitor()
        mon.register_models("m", "f")
        for i in range(TTFTMonitor.HISTORY_SIZE + 20):
            mon.record("m", float(i))
        assert len(mon.smart_ttft) == TTFTMonitor.HISTORY_SIZE
        assert mon.smart_ttft[0] == 20.0

    def test_multiple_models_independent(self):
        mon = TTFTMonitor()
        mon.register_models("smart", "fast")
        mon.record("smart", 1.0)
        mon.record("fast", 0.1)
        mon.record("other", 9.9)
        assert list(mon.smart_ttft) == [1.0]
        assert list(mon.fast_ttft) == [0.1]


class TestShorten:
    @pytest.mark.parametrize("model,expected", [
        ("claude-3-opus-20240229", "3-opus"),
        ("claude-3-haiku-20240307", "3-haiku"),
        ("claude-3-5-sonnet-20241022", "3-5-sonnet"),
        ("gpt-4", "gpt-4"),
        ("claude-opus", "opus"),
    ])
    def test_shorten(self, model, expected):
        assert TTFTMonitor.shorten(model) == expected

class TestCachePct:
    def test_a_hosted_and_a_local_model_share_one_series_in_request_order(self):
        mon = TTFTMonitor()
        mon.record_cache_pct("claude-opus", 42)
        mon.record_cache_pct("deepseek-v4", 88)
        assert list(mon.cache_samples) == [("claude-opus", 42), ("deepseek-v4", 88)]

    def test_cache_samples_capped_at_history_size(self):
        mon = TTFTMonitor()
        for i in range(TTFTMonitor.HISTORY_SIZE + 10):
            mon.record_cache_pct("m", i)
        assert len(mon.cache_samples) == TTFTMonitor.HISTORY_SIZE
        assert mon.cache_samples[0] == ("m", 10)


class TestSamples:
    def test_samples_keep_every_model_in_the_order_recorded(self):
        mon = TTFTMonitor()
        mon.register_models("smart", "fast")
        mon.record("smart", 1.0)
        mon.record("fast", 0.1)
        mon.record("smart", 1.2)
        assert list(mon.samples) == [("smart", 1.0), ("fast", 0.1), ("smart", 1.2)]

    def test_samples_are_capped_like_the_per_model_lists(self):
        mon = TTFTMonitor()
        for i in range(TTFTMonitor.HISTORY_SIZE + 5):
            mon.record("a" if i % 2 else "b", float(i))
        assert len(mon.samples) == TTFTMonitor.HISTORY_SIZE
        assert mon.samples[0] == ("a", 5.0)
