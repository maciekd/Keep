"""AbstractLLM.record_cache_hit — the one entry the dashboard's live KV-cache chart and its
per-day log scan are fed through, by every backend that knows its hit."""
from unittest.mock import Mock, patch

import pytest

from assistant.llm.AbstractLLM import AbstractLLM


def record(cached: int, total: int, model: str = "deepseek-v4", **kwargs):
    """(the samples the chart got, the trace lines written)"""
    llm = Mock(spec=AbstractLLM)
    llm.name.return_value, llm.model = "DS4", model
    written = []
    with patch("assistant.llm.AbstractLLM.trace", written.append), \
            patch("common.TTFTMonitor.TTFTMonitor.get_instance") as monitor:
        AbstractLLM.record_cache_hit(llm, cached, total, **kwargs)
    samples = [call.args for call in monitor.return_value.record_cache_pct.call_args_list]
    return samples, written


@pytest.mark.unit
class TestRecordCacheHit:
    def test_the_sample_is_the_rounded_share_under_the_models_name(self):
        # the chart colours a column by its model, so the sample must carry the instance's, not the backend's
        samples, _ = record(2, 3, model="deepseek-v4")
        assert samples == [("deepseek-v4", 67)]

    def test_a_request_without_prompt_tokens_is_no_sample(self):
        # a division by zero here would fail the request a diagnostic rides on
        samples, written = record(0, 0)
        assert samples == []
        assert not any(" read," in line for line in written), "nothing for the log scan either"

    def test_a_backend_with_a_line_of_its_own_adds_no_second_one(self):
        # Claude logs the richer line itself; a second would count the request twice in the day's median
        samples, written = record(900, 1000, log=False)
        assert samples == [("deepseek-v4", 90)]
        assert not any(" read," in line for line in written)
