"""What several completions used together, and the log line of a call."""
from unittest.mock import Mock

import pytest

from assistant.llm.LLMUsage import LLMUsage


class TestSum:
    def test_tokens_and_costs_add_up(self):
        total = (LLMUsage(input_tokens=100, cache_write_5m=10, cache_write_1h=20, cache_read=1000,
                          output_tokens=50, cost_usd=0.25)
                 + LLMUsage(input_tokens=1, cache_write_5m=2, cache_write_1h=3, cache_read=4,
                            output_tokens=5, cost_usd=0.5))
        assert (total.input_tokens, total.cache_write_5m, total.cache_write_1h, total.cache_read,
                total.output_tokens) == (101, 12, 23, 1004, 55)
        assert total.cost_usd == pytest.approx(0.75)

    @pytest.mark.parametrize("costs", [(None, 0.5), (0.5, None)])
    def test_one_call_without_a_price_makes_the_whole_sum_unpriced(self, costs):
        """A sum that counted the unpriced call as free would understate the pass."""
        total = LLMUsage(output_tokens=1, cost_usd=costs[0]) + LLMUsage(output_tokens=2, cost_usd=costs[1])
        assert total.cost_usd is None
        assert total.output_tokens == 3
        assert "$" not in total.describe()

    def test_a_sum_is_at_batch_price_only_when_every_call_was(self):
        batched, live = LLMUsage(batch=True, cost_usd=1.0), LLMUsage(cost_usd=1.0)
        assert (batched + batched).batch
        assert not (batched + live).batch and not (live + batched).batch
        assert "batch" not in (batched + live).describe()


class TestDescribeCall:
    def test_a_backend_that_reports_neither_says_so_instead_of_inventing_zeros(self):
        line = LLMUsage.describe_call(Mock(spec=[]))
        assert "0 " not in line and "$" not in line

    def test_a_mocked_backend_is_not_taken_for_one_that_reports_usage(self):
        """`of` is asked of whatever stands in for an LLM, a wrapper's `__getattr__` included:
        only a real usage counts."""
        assert LLMUsage.of(Mock()) is None

    def test_the_line_carries_the_stop_reason_the_cache_tokens_and_the_cost(self):
        llm = Mock(last_stop_reason="max_tokens",
                   last_usage=LLMUsage(input_tokens=12, cache_write_5m=300, cache_write_1h=4000,
                                       cache_read=50000, output_tokens=678, cost_usd=1.239))
        line = LLMUsage.describe_call(llm)
        assert "max_tokens" in line
        assert "4300 cache-written" in line and "50000 cache-read" in line and "678 output" in line
        assert "$1.24" in line
