import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from common.logging import fine, trace
from .ConfigManager import config


class AnthropicCostMonitor:
    """Periodically fetches Anthropic organization spend per API key into the DashboardTab.

    Requires the `api_keys.anthropic_admin` key; runs on a daemon thread and reports
    fetch problems through the tab's cost_error rather than raising. The figures are the
    billed ones of the Usage and Cost Analytics API (beta), which can lag the usage by some
    hours; spend made without an API key (the Console's workbench) is shown as "No key".
    """

    API_URL = "https://api.anthropic.com/v1/organizations/analytics/cost"
    KEYS_URL = "https://api.anthropic.com/v1/organizations/api_keys"
    BETA = "analytics-usage-cost-2026-08-24"
    NO_KEY = "No key"
    MIN_KEY_COST = 1.0  # Only show keys with > $1 total spend
    MAX_PAGES = 10
    MAX_DAYS = 60
    API_SPAN_DAYS = 31  # the most one request may cover
    API_PAGE_LIMIT = 1000

    def __init__(self, dashboard_tab, fetch_interval: float = 300.0):
        self._dashboard_tab = dashboard_tab
        self._fetch_interval = fetch_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread and self._thread.is_alive() and not self._stop_event.is_set():
            return
        # A stopped loop still mid-sample keeps the event it was started with and ends on it;
        # this start gets a fresh one. With one event shared, coming back to the tab before
        # the old loop had ended started nothing and left the monitor dead while the tab showed.
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor_loop, args=(self._stop_event,), daemon=True,
                                        name="AnthropicCostMonitor")
        self._thread.start()
        fine("AnthropicCostMonitor started")

    def stop(self):
        """Returns at once — a tab switch calls it on the TUI's input thread, and a join here
        held every keypress back while the loop was mid-sample. The loop ends by itself at
        its next look at its event."""
        self._stop_event.set()

    def _monitor_loop(self, stop_event: threading.Event):
        while not stop_event.is_set():
            try:
                self._fetch_cost_data()
            except Exception as e:
                trace(f"AnthropicCostMonitor error: {e}")
                self._dashboard_tab.cost_error = str(e)
            stop_event.wait(self._fetch_interval)

    def _fetch_cost_data(self):
        api_key = config.get('api_keys.anthropic_admin', '')
        if not api_key:
            self._dashboard_tab.cost_error = "Configure 'api_keys.anthropic_admin' in config.yaml"
            trace("AnthropicCostMonitor: no admin API key configured")
            return

        today = datetime.now(timezone.utc).date()
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": self.BETA,
        }

        # One row per day and key that spent anything, fetched a span at a time up to the end of today
        rows = []
        span_start = today - timedelta(days=self.MAX_DAYS - 1)
        end = today + timedelta(days=1)
        while span_start < end:
            span_end = min(span_start + timedelta(days=self.API_SPAN_DAYS), end)
            span_rows = self._fetch_rows(headers, span_start, span_end)
            if span_rows is None:
                return
            rows.extend(span_rows)
            span_start = span_end

        key_ids = {row.get("group", {}).get("api_key_id") for row in rows}
        names = self._key_names(headers) if key_ids - {None} else {}

        # Aggregate: {date_str: {key name: total_dollars}}; amount is cents as a decimal string
        daily_costs = defaultdict(lambda: defaultdict(float))
        key_totals = defaultdict(float)

        for row in rows:
            date_str = row.get("starting_at", "")[:10]  # "2026-04-15"
            key_id = row.get("group", {}).get("api_key_id")
            key = names.get(key_id, key_id) if key_id else self.NO_KEY
            amount_dollars = float(row.get("cost", {}).get("amount", "0")) / 100.0
            daily_costs[date_str][key] += amount_dollars
            key_totals[key] += amount_dollars

        significant = sorted(k for k, total in key_totals.items() if total > self.MIN_KEY_COST)
        if len(significant) < len(key_totals):
            trace(f"AnthropicCostMonitor: left out keys at <= ${self.MIN_KEY_COST:.0f}: "
                  f"{sorted(set(key_totals) - set(significant))}")

        filtered_daily = {}
        for date_str, keys in daily_costs.items():
            filtered = {k: cost for k, cost in keys.items() if k in significant}
            if filtered:
                filtered_daily[date_str] = filtered

        cost_dates = [(today - timedelta(days=self.MAX_DAYS - 1 - i)) for i in range(self.MAX_DAYS)]
        cost_date_strs = [d.strftime("%Y-%m-%d") for d in cost_dates]

        thirty_day_strs = set(cost_date_strs[-30:])
        today_str = {cost_date_strs[-1]}
        yesterday_str = {cost_date_strs[-2]} if len(cost_date_strs) >= 2 else set()

        def _sum_costs(date_set):
            return sum(sum(models.values()) for ds, models in filtered_daily.items() if ds in date_set)

        self._dashboard_tab.daily_costs = filtered_daily
        self._dashboard_tab.cost_keys = significant
        self._dashboard_tab.total_cost_30d = _sum_costs(thirty_day_strs)
        self._dashboard_tab.total_cost_1d = _sum_costs(today_str)
        self._dashboard_tab.total_cost_yesterday = _sum_costs(yesterday_str)
        self._dashboard_tab.cost_dates = cost_dates
        self._dashboard_tab.cost_date_strs = cost_date_strs
        self._dashboard_tab.cost_error = ""
        self._dashboard_tab.cost_last_updated = time.time()

        fine(f"AnthropicCostMonitor: fetched {len(rows)} rows, "
             f"{len(significant)} keys > $1 {significant}, "
             f"30d ${self._dashboard_tab.total_cost_30d:.0f} / "
             f"yesterday ${self._dashboard_tab.total_cost_yesterday:.0f} / 1d ${self._dashboard_tab.total_cost_1d:.0f}")

    def _fetch_rows(self, headers: dict, start, end) -> Optional[list]:
        """The cost rows of the days from `start` up to `end`, every page of them; None when
        the admin key is refused, which the tab is told."""
        params = {
            "starting_at": start.strftime("%Y-%m-%dT00:00:00Z"),
            "ending_at": end.strftime("%Y-%m-%dT00:00:00Z"),
            "time_bucket_width": "1d",
            "group_by[]": "api_key_id",
            "limit": self.API_PAGE_LIMIT,
        }
        rows = []
        for _ in range(self.MAX_PAGES):
            resp = requests.get(self.API_URL, headers=headers, params=params, timeout=30)

            if resp.status_code in (401, 403):
                self._dashboard_tab.cost_error = "Invalid admin API key"
                trace(f"AnthropicCostMonitor: auth failed ({resp.status_code})")
                return None

            if resp.status_code >= 400:
                trace(f"AnthropicCostMonitor: API error {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            body = resp.json()
            rows.extend(body.get("data", []))

            if not body.get("has_more"):
                break
            params["page"] = body.get("next_page")
        return rows

    def _key_names(self, headers: dict) -> dict:
        """API key id -> name, inactive and archived ones included so their past spend keeps
        its name; {} when the list cannot be read, and the chart shows the ids instead."""
        try:
            resp = requests.get(self.KEYS_URL, headers=headers, timeout=30, params={"limit": 1000})
            resp.raise_for_status()
            names = {k["id"]: k["name"] for k in resp.json().get("data", [])}
            trace(f"AnthropicCostMonitor: {len(names)} keys named")
            return names
        except Exception as e:
            trace(f"AnthropicCostMonitor: key names unavailable, showing ids: {e}")
            return {}
