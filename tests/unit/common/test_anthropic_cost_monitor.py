from unittest.mock import Mock, patch, MagicMock

import pytest

from common.AnthropicCostMonitor import AnthropicCostMonitor


def _make_monitor():
    tab = Mock()
    tab.cost_error = ""
    tab.daily_costs = {}
    tab.cost_workspaces = []
    tab.total_cost_30d = 0.0
    tab.total_cost_7d = 0.0
    tab.total_cost_1d = 0.0
    tab.cost_dates = []
    tab.cost_date_strs = []
    tab.cost_last_updated = 0.0
    return AnthropicCostMonitor(tab, fetch_interval=0), tab


def _bucket(date_str, results):
    """Build one API response bucket. amounts are in cents; a None workspace id is the default workspace."""
    return {
        "starting_at": f"{date_str}T00:00:00Z",
        "results": [
            {"workspace_id": workspace_id, "amount": str(cents)}
            for workspace_id, cents in results
        ],
    }


def _routed_get(cost_body, workspaces_body=None, workspaces_status=200):
    """A requests.get that answers the cost report and the workspace list by URL."""
    def get(url, **kwargs):
        if url == AnthropicCostMonitor.WORKSPACES_URL:
            resp = MagicMock(status_code=workspaces_status, json=lambda: workspaces_body)
            if workspaces_status >= 400:
                resp.raise_for_status.side_effect = Exception(f"HTTP {workspaces_status}")
            return resp
        return MagicMock(status_code=200, json=lambda: cost_body)
    return get


@pytest.mark.unit
class TestAnthropicCostMonitorAggregation:

    @patch('common.AnthropicCostMonitor.config')
    @patch('common.AnthropicCostMonitor.requests.get')
    def test_aggregates_daily_costs_cents_to_dollars(self, mock_get, mock_config):
        mock_config.get.return_value = "test-key"
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "data": [
                    _bucket("2026-04-20", [(None, "5000")]),
                    _bucket("2026-04-19", [(None, "3000")]),
                ],
                "has_more": False,
            }
        )

        monitor, tab = _make_monitor()
        monitor._fetch_cost_data()

        assert tab.daily_costs["2026-04-20"]["Default"] == 50.0
        assert tab.daily_costs["2026-04-19"]["Default"] == 30.0
        assert tab.cost_error == ""

    @patch('common.AnthropicCostMonitor.config')
    @patch('common.AnthropicCostMonitor.requests.get')
    def test_filters_workspaces_below_min_cost(self, mock_get, mock_config):
        mock_config.get.return_value = "test-key"
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "data": [
                    _bucket("2026-04-20", [
                        (None, "5000"),                # $50 total -> kept
                        ("wrkspc_cheap", "50"),         # $0.50 total -> filtered
                    ]),
                ],
                "has_more": False,
            }
        )

        monitor, tab = _make_monitor()
        monitor._fetch_cost_data()

        assert "Default" in tab.cost_workspaces
        assert "wrkspc_cheap" not in tab.cost_workspaces
        assert "wrkspc_cheap" not in tab.daily_costs.get("2026-04-20", {})

    @patch('common.AnthropicCostMonitor.config')
    @patch('common.AnthropicCostMonitor.requests.get')
    def test_aggregates_same_workspace_across_buckets(self, mock_get, mock_config):
        mock_config.get.return_value = "test-key"
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "data": [
                    _bucket("2026-04-20", [(None, "2000")]),
                    _bucket("2026-04-20", [(None, "3000")]),
                ],
                "has_more": False,
            }
        )

        monitor, tab = _make_monitor()
        monitor._fetch_cost_data()

        assert tab.daily_costs["2026-04-20"]["Default"] == 50.0

    @patch('common.AnthropicCostMonitor.config')
    @patch('common.AnthropicCostMonitor.requests.get')
    def test_handles_pagination(self, mock_get, mock_config):
        mock_config.get.return_value = "test-key"
        mock_get.side_effect = [
            MagicMock(
                status_code=200,
                json=lambda: {
                    "data": [_bucket("2026-04-20", [(None, "5000")])],
                    "has_more": True,
                    "next_page": "page2",
                }
            ),
            MagicMock(
                status_code=200,
                json=lambda: {
                    "data": [_bucket("2026-04-19", [(None, "3000")])],
                    "has_more": False,
                }
            ),
        ]

        monitor, tab = _make_monitor()
        monitor._fetch_cost_data()

        assert "2026-04-20" in tab.daily_costs
        assert "2026-04-19" in tab.daily_costs
        assert mock_get.call_count == 2

    @patch('common.AnthropicCostMonitor.config')
    @patch('common.AnthropicCostMonitor.requests.get')
    def test_30_day_total_excludes_older_data(self, mock_get, mock_config):
        mock_config.get.return_value = "test-key"
        # Create a bucket 45 days ago — outside the 30-day window
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        old_date = (now - timedelta(days=45)).strftime("%Y-%m-%d")
        recent_date = now.strftime("%Y-%m-%d")

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "data": [
                    _bucket(old_date, [(None, "10000")]),
                    _bucket(recent_date, [(None, "5000")]),
                ],
                "has_more": False,
            }
        )

        monitor, tab = _make_monitor()
        monitor._fetch_cost_data()

        # 30-day total should only include recent_date's $50
        assert tab.total_cost_30d == 50.0

    @patch('common.AnthropicCostMonitor.config')
    @patch('common.AnthropicCostMonitor.requests.get')
    def test_7d_and_1d_totals(self, mock_get, mock_config):
        mock_config.get.return_value = "test-key"
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        three_days_ago = (now - timedelta(days=3)).strftime("%Y-%m-%d")
        ten_days_ago = (now - timedelta(days=10)).strftime("%Y-%m-%d")

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "data": [
                    _bucket(today, [(None, "2000")]),          # $20 - in 1d, 7d, 30d
                    _bucket(three_days_ago, [(None, "3000")]), # $30 - in 7d, 30d
                    _bucket(ten_days_ago, [(None, "5000")]),   # $50 - in 30d only
                ],
                "has_more": False,
            }
        )

        monitor, tab = _make_monitor()
        monitor._fetch_cost_data()

        assert tab.total_cost_1d == 20.0
        assert tab.total_cost_7d == 50.0
        assert tab.total_cost_30d == 100.0

    @patch('common.AnthropicCostMonitor.config')
    @patch('common.AnthropicCostMonitor.requests.get')
    def test_auth_failure_sets_error(self, mock_get, mock_config):
        mock_config.get.return_value = "bad-key"
        mock_get.return_value = MagicMock(status_code=401)

        monitor, tab = _make_monitor()
        monitor._fetch_cost_data()

        assert "Invalid admin API key" in tab.cost_error

    @patch('common.AnthropicCostMonitor.config')
    @patch('common.AnthropicCostMonitor.requests.get')
    def test_empty_response_no_crash(self, mock_get, mock_config):
        mock_config.get.return_value = "test-key"
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [], "has_more": False}
        )

        monitor, tab = _make_monitor()
        monitor._fetch_cost_data()

        assert tab.cost_workspaces == []
        assert tab.total_cost_30d == 0.0
        assert tab.total_cost_7d == 0.0
        assert tab.total_cost_1d == 0.0
        assert tab.cost_error == ""

    @patch('common.AnthropicCostMonitor.config')
    @patch('common.AnthropicCostMonitor.requests.get')
    def test_workspaces_are_shown_by_name(self, mock_get, mock_config):
        mock_config.get.return_value = "test-key"
        mock_get.side_effect = _routed_get(
            {"data": [_bucket("2026-04-20", [(None, "4000"), ("wrkspc_dev", "2000"), ("wrkspc_gone", "300")])],
             "has_more": False},
            {"data": [{"id": "wrkspc_dev", "name": "Dev"}], "has_more": False})

        monitor, tab = _make_monitor()
        monitor._fetch_cost_data()

        assert tab.daily_costs["2026-04-20"] == {"Default": 40.0, "Dev": 20.0, "wrkspc_gone": 3.0}
        assert tab.cost_workspaces == ["Default", "Dev", "wrkspc_gone"]

    @patch('common.AnthropicCostMonitor.config')
    @patch('common.AnthropicCostMonitor.requests.get')
    def test_unreadable_workspace_list_falls_back_to_ids(self, mock_get, mock_config):
        mock_config.get.return_value = "test-key"
        mock_get.side_effect = _routed_get(
            {"data": [_bucket("2026-04-20", [("wrkspc_dev", "2000")])], "has_more": False},
            workspaces_status=500)

        monitor, tab = _make_monitor()
        monitor._fetch_cost_data()

        assert tab.daily_costs["2026-04-20"] == {"wrkspc_dev": 20.0}
        assert tab.cost_error == ""

    @patch('common.AnthropicCostMonitor.config')
    @patch('common.AnthropicCostMonitor.requests.get')
    def test_only_the_default_workspace_skips_the_name_lookup(self, mock_get, mock_config):
        mock_config.get.return_value = "test-key"
        mock_get.side_effect = _routed_get({"data": [_bucket("2026-04-20", [(None, "2000")])], "has_more": False})

        monitor, tab = _make_monitor()
        monitor._fetch_cost_data()

        assert [c.args[0] for c in mock_get.call_args_list] == [AnthropicCostMonitor.API_URL]


@pytest.mark.unit
class TestWorkspaceSeries:
    @patch('common.AnthropicCostMonitor.config')
    @patch('common.AnthropicCostMonitor.requests.get')
    def test_two_workspace_ids_with_one_name_are_one_series_before_the_threshold(self, mock_get, mock_config):
        # An archived workspace and its successor carry one name: their spend is one series,
        # and it is the series' total — not each id's — that the $1 threshold weighs.
        mock_config.get.return_value = "test-key"
        mock_get.side_effect = _routed_get(
            {"data": [_bucket("2026-04-20", [("wrkspc_a", "60"), ("wrkspc_b", "60")])], "has_more": False},
            {"data": [{"id": "wrkspc_a", "name": "Dev"}, {"id": "wrkspc_b", "name": "Dev"}], "has_more": False})

        monitor, tab = _make_monitor()
        monitor._fetch_cost_data()

        assert tab.daily_costs["2026-04-20"] == {"Dev": pytest.approx(1.2)}
        assert tab.cost_workspaces == ["Dev"]
