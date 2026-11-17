from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch
from xml.etree import ElementTree

import pytest
import requests

from integrations.ferry import FerryIntegration as module
from integrations.ferry.FerryIntegration import FerryError, FerryIntegration, matches, parse, render, request_body

CET = timezone(timedelta(hours=1))
NOW = datetime(2026, 11, 17, 14, 2, tzinfo=CET)


def announcement(clock: str, origin: str, destination: str, **extra) -> dict:
    return {"DepartureTime": f"2026-11-17T{clock}:00.000+01:00", "FromHarbor": {"Name": origin},
            "ToHarbor": {"Name": destination}, "Deleted": False, **extra}


def answer(*items) -> dict:
    return {"RESPONSE": {"RESULT": [{"FerryAnnouncement": list(items)}]}}


TIMETABLE = answer(
    announcement("14:25", "Lilla Varholmen", "Hönö"),
    announcement("14:10", "Hönö", "Lilla Varholmen"),
    announcement("14:40", "Hönö", "Lilla Varholmen", Info=["Reduced capacity: one vessel in service"]),
    announcement("15:30", "Hönö", "Lilla Varholmen"),
    announcement("14:55", "Lilla Varholmen", "Hönö", Deleted=True),
)


class TestRequest:
    def test_the_key_and_route_are_escaped_into_the_xml(self):
        root = ElementTree.fromstring(request_body('k"<ey', 'Hö"nö'))
        assert root.find("LOGIN").get("authenticationkey") == 'k"<ey'
        assert root.find(".//EQ[@name='Route.Name']").get("value") == 'Hö"nö'

    def test_the_query_asks_for_departures_still_ahead_in_order(self):
        body = request_body("key", "Hönöleden")
        assert '<GT name="DepartureTime" value="$now"/>' in body
        assert 'orderby="DepartureTime"' in body


class TestParse:
    def test_announcements_are_sorted_and_deleted_ones_dropped(self):
        departures = parse(TIMETABLE)
        assert [d.time.strftime("%H:%M") for d in departures] == ["14:10", "14:25", "14:40", "15:30"]

    def test_an_api_error_becomes_a_ferry_error_with_its_message(self):
        with pytest.raises(FerryError, match="Invalid authentication"):
            parse({"RESPONSE": {"RESULT": [{"ERROR": {"MESSAGE": "Invalid authentication key"}}]}})

    def test_an_announcement_without_harbors_is_skipped(self):
        departures = parse(answer({"DepartureTime": "2026-11-17T14:10:00+01:00"}, announcement("14:20", "Hönö", "Lilla Varholmen")))
        assert len(departures) == 1


class TestMatching:
    @pytest.mark.parametrize("asked", ["Hönö", "hono", "HÖNÖ PINAN"])
    def test_a_loosely_named_harbor_matches(self, asked):
        assert matches("Hönö Pinan", asked)

    def test_another_harbor_does_not(self):
        assert not matches("Lilla Varholmen", "Hönö")


class TestRender:
    def test_both_directions_by_default_with_minutes_for_the_near_ones(self):
        text = render("Hönöleden", parse(TIMETABLE), NOW)
        hono = next(line for line in text.splitlines() if line.startswith("From Hönö"))
        varholmen = next(line for line in text.splitlines() if line.startswith("From Lilla Varholmen"))
        assert "14:10 (in 8 min)" in hono and "15:30" in hono and "(in 88" not in hono
        assert "14:25 (in 23 min)" in varholmen

    def test_one_side_only_when_an_origin_is_named(self):
        text = render("Hönöleden", parse(TIMETABLE), NOW, origin="varholmen")
        assert "From Lilla Varholmen" in text and "From Hönö" not in text

    def test_notices_of_shown_departures_are_listed(self):
        assert "Reduced capacity" in render("Hönöleden", parse(TIMETABLE), NOW)

    def test_the_count_limits_each_direction(self):
        text = render("Hönöleden", parse(TIMETABLE), NOW, origin="Hönö", count=1)
        assert "14:10" in text and "14:40" not in text

    def test_an_unknown_harbor_lists_the_known_ones(self):
        text = render("Hönöleden", parse(TIMETABLE), NOW, origin="Göteborg")
        assert "Lilla Varholmen" in text and "From" not in text

    def test_a_departure_tomorrow_carries_its_day(self):
        late = answer({**announcement("00:00", "Hönö", "Lilla Varholmen"), "DepartureTime": "2026-11-18T06:00:00+01:00"})
        assert "Wed 06:00" in render("Hönöleden", parse(late), NOW)


class TestFetch:
    @pytest.fixture
    def post(self):
        with patch.object(module.config, "get", side_effect=lambda key, default=None:
                          {"ferry.api_key": "key", "ferry.route": "Hönöleden"}.get(key, default)), \
             patch.object(module.requests, "post") as post:
            yield post

    def test_a_refused_key_says_which_config_key(self, post):
        post.return_value = Mock(status_code=401)
        with pytest.raises(FerryError, match="ferry.api_key"):
            FerryIntegration().fetch()

    def test_a_timeout_becomes_a_ferry_error(self, post):
        post.side_effect = requests.Timeout()
        with pytest.raises(FerryError):
            FerryIntegration().fetch()

    def test_the_key_travels_in_the_body_not_the_url(self, post):
        post.return_value = Mock(status_code=200, json=Mock(return_value=TIMETABLE))
        FerryIntegration().fetch()
        url = post.call_args.args[0]
        assert "key" not in url.split("/")[-1] and b'authenticationkey="key"' in post.call_args.kwargs["data"]

    def test_health_fails_when_the_route_has_no_departures(self, post):
        post.return_value = Mock(status_code=200, json=Mock(return_value=answer()))
        with pytest.raises(FerryError, match="ferry.route"):
            FerryIntegration().check_health()
