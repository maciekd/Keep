from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock

import pytest

from common.AnthropicCostMonitor import AnthropicCostMonitor


def _make_monitor():
    tab = Mock()
    tab.cost_error = ""
    tab.daily_costs = {}
    tab.cost_keys = []
    tab.total_cost_30d = 0.0
    tab.total_cost_1d = 0.0
    tab.cost_dates = []
    tab.cost_date_strs = []
    tab.cost_last_updated = 0.0
    return AnthropicCostMonitor(tab, fetch_interval=0), tab


def _day(days_ago=0):
    return (datetime.now(timezone.utc).date() - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def _row(date_str, key_id, cents):
    """One row of the cost analytics: a day, a key (None: spend made without one), cents."""
    return {"starting_at": f"{date_str}T00:00:00Z", "group": {"api_key_id": key_id},
            "cost": {"amount": str(cents)}}


class _Api:
    """A requests.get standing in for the two endpoints: the key list by URL, and the cost
    rows whose day lies in the span a request asks for, `page_size` rows to a page."""

    def __init__(self, rows, keys=None, keys_status=200, page_size=None):
        self.rows, self.keys, self.keys_status, self.page_size = rows, keys or [], keys_status, page_size
        self.calls = []

    def __call__(self, url, headers=None, params=None, **kwargs):
        self.calls.append((url, dict(headers or {}), dict(params or {})))
        if url == AnthropicCostMonitor.KEYS_URL:
            resp = MagicMock(status_code=self.keys_status, json=lambda: {"data": self.keys, "has_more": False})
            if self.keys_status >= 400:
                resp.raise_for_status.side_effect = Exception(f"HTTP {self.keys_status}")
            return resp
        in_span = [r for r in self.rows if params["starting_at"] <= r["starting_at"] < params["ending_at"]]
        size = self.page_size or len(in_span) or 1
        start = int(params.get("page", 0))
        body = {"data": in_span[start:start + size], "has_more": start + size < len(in_span)}
        if body["has_more"]:
            body["next_page"] = str(start + size)
        return MagicMock(status_code=200, json=lambda: body)

    def cost_calls(self):
        return [(headers, params) for url, headers, params in self.calls if url == AnthropicCostMonitor.API_URL]


def _fetch(api):
    monitor, tab = _make_monitor()
    with patch('common.AnthropicCostMonitor.config') as config, \
            patch('common.AnthropicCostMonitor.requests.get', side_effect=api):
        config.get.return_value = "test-key"
        monitor._fetch_cost_data()
    return tab


@pytest.mark.unit
class TestAnthropicCostMonitorAggregation:

    def test_aggregates_daily_costs_cents_to_dollars(self):
        tab = _fetch(_Api([_row(_day(0), None, "5000"), _row(_day(1), None, "3000.500000")]))

        assert tab.daily_costs[_day(0)] == {"No key": 50.0}
        assert tab.daily_costs[_day(1)] == {"No key": pytest.approx(30.005)}
        assert tab.cost_error == ""

    def test_filters_keys_below_min_cost(self):
        tab = _fetch(_Api([_row(_day(0), "apikey_big", "5000"), _row(_day(0), "apikey_cheap", "50")]))

        assert tab.cost_keys == ["apikey_big"]
        assert "apikey_cheap" not in tab.daily_costs[_day(0)]

    def test_a_key_under_the_threshold_on_each_day_is_kept_when_its_total_is_over(self):
        tab = _fetch(_Api([_row(_day(0), "apikey_slow", "60"), _row(_day(1), "apikey_slow", "60")]))

        assert tab.cost_keys == ["apikey_slow"]

    def test_follows_the_pages_of_a_span(self):
        api = _Api([_row(_day(0), "apikey_a", "5000"), _row(_day(1), "apikey_a", "3000"),
                    _row(_day(2), "apikey_a", "2000")], page_size=1)
        tab = _fetch(api)

        assert {d: c["apikey_a"] for d, c in tab.daily_costs.items()} == {_day(0): 50.0, _day(1): 30.0, _day(2): 20.0}

    def test_the_chart_days_are_asked_for_once_each_in_spans_the_api_accepts(self):
        # The analytics API refuses a request spanning more than 31 days.
        api = _Api([])
        tab = _fetch(api)

        spans = [(datetime.fromisoformat(p["starting_at"].replace("Z", "+00:00")),
                  datetime.fromisoformat(p["ending_at"].replace("Z", "+00:00"))) for _, p in api.cost_calls()]
        assert all(timedelta(0) < end - start <= timedelta(days=31) for start, end in spans)
        assert all(prev_end == start for (_, prev_end), (start, _) in zip(spans, spans[1:])), "no gap, no overlap"
        assert spans[0][0].strftime("%Y-%m-%d") == tab.cost_date_strs[0]
        assert spans[-1][1].strftime("%Y-%m-%d") == (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat(), \
            "today's bucket is inside the last span"

    def test_a_day_in_both_halves_of_the_range_is_counted_once(self):
        rows = [_row(_day(n), "apikey_a", "1000") for n in range(60)]
        tab = _fetch(_Api(rows))

        assert len(tab.daily_costs) == 60
        assert all(costs == {"apikey_a": 10.0} for costs in tab.daily_costs.values())

    def test_the_request_is_the_per_key_daily_report_of_the_beta_analytics_api(self):
        # A contract with Anthropic's API: without the beta header the endpoint answers 404,
        # and without the bucket width a row is the whole range.
        api = _Api([])
        _fetch(api)

        headers, params = api.cost_calls()[0]
        assert headers["anthropic-beta"] == "analytics-usage-cost-2026-08-24"
        assert params["group_by[]"] == "api_key_id"
        assert params["time_bucket_width"] == "1d"

    def test_30_day_total_excludes_older_data(self):
        tab = _fetch(_Api([_row(_day(45), None, "10000"), _row(_day(0), None, "5000")]))

        assert tab.total_cost_30d == 50.0

    def test_today_and_yesterday_totals(self):
        tab = _fetch(_Api([_row(_day(0), None, "2000"), _row(_day(1), None, "700"), _row(_day(1), "apikey_a", "800"),
                           _row(_day(10), None, "5000")]))

        assert tab.total_cost_1d == 20.0
        assert tab.total_cost_yesterday == 15.0
        assert tab.total_cost_30d == 85.0

    @patch('common.AnthropicCostMonitor.config')
    @patch('common.AnthropicCostMonitor.requests.get')
    def test_auth_failure_sets_error(self, mock_get, mock_config):
        mock_config.get.return_value = "bad-key"
        mock_get.return_value = MagicMock(status_code=401)

        monitor, tab = _make_monitor()
        monitor._fetch_cost_data()

        assert "admin API key" in tab.cost_error
        assert tab.daily_costs == {}, "nothing is published from a refused fetch"

    def test_empty_response_no_crash(self):
        tab = _fetch(_Api([]))

        assert tab.cost_keys == []
        assert tab.total_cost_30d == 0.0
        assert tab.total_cost_1d == 0.0
        assert tab.cost_error == ""


@pytest.mark.unit
class TestKeySeries:

    def test_keys_are_shown_by_name(self):
        tab = _fetch(_Api([_row(_day(0), None, "4000"), _row(_day(0), "apikey_dev", "2000"),
                           _row(_day(0), "apikey_gone", "300")],
                          keys=[{"id": "apikey_dev", "name": "marcus-dev"}]))

        assert tab.daily_costs[_day(0)] == {"No key": 40.0, "marcus-dev": 20.0, "apikey_gone": 3.0}
        assert tab.cost_keys == ["No key", "apikey_gone", "marcus-dev"]

    def test_unreadable_key_list_falls_back_to_ids(self):
        tab = _fetch(_Api([_row(_day(0), "apikey_dev", "2000")], keys_status=500))

        assert tab.daily_costs[_day(0)] == {"apikey_dev": 20.0}
        assert tab.cost_error == ""

    def test_spend_without_any_key_skips_the_name_lookup(self):
        api = _Api([_row(_day(0), None, "2000")])
        _fetch(api)

        assert AnthropicCostMonitor.KEYS_URL not in [url for url, _, _ in api.calls]

    def test_two_key_ids_with_one_name_are_one_series_before_the_threshold(self):
        # Keys of two workspaces can carry one name: their spend is one series, and it is
        # the series' total — not each id's — that the $1 threshold weighs.
        tab = _fetch(_Api([_row(_day(0), "apikey_a", "60"), _row(_day(0), "apikey_b", "60")],
                          keys=[{"id": "apikey_a", "name": "dev"}, {"id": "apikey_b", "name": "dev"}]))

        assert tab.daily_costs[_day(0)] == {"dev": pytest.approx(1.2)}
        assert tab.cost_keys == ["dev"]
