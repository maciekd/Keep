from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch
from xml.etree import ElementTree

import pytest
import requests

from integrations.ferry import FerryIntegration as module
from integrations.ferry.FerryIntegration import (FerryError, FerryIntegration, closest, matches, parse, parse_routes,
                                                 render, request_body, resolve)

CET = timezone(timedelta(hours=1))
ROUTE = "Ekeleden"
NOW = datetime(2026, 11, 17, 14, 2, tzinfo=CET)


def announcement(clock: str, origin: str, destination: str, **extra) -> dict:
    return {"DepartureTime": f"2026-11-17T{clock}:00.000+01:00", "FromHarbor": {"Name": origin},
            "ToHarbor": {"Name": destination}, "Deleted": False, **extra}


def answer(*items) -> dict:
    return {"RESPONSE": {"RESULT": [{"FerryAnnouncement": list(items)}]}}


def route_list(*names, deleted=()) -> dict:
    return {"RESPONSE": {"RESULT": [{"FerryRoute": [{"Name": n, "Deleted": False} for n in names]
                                     + [{"Name": n, "Deleted": True} for n in deleted]}]}}


ROUTES = route_list("Ekeleden", "Norrleden", "Strömsleden", "Älvleden", "Bodaleden")


def api(departures: dict, routes: dict = ROUTES):
    """A fake `requests.post` answering the route list and the departures by the query's object type."""
    def post(url, data, **kwargs):
        payload = routes if b'objecttype="FerryRoute"' in data else departures
        return Mock(status_code=200, json=Mock(return_value=payload))
    return post


TIMETABLE = answer(
    announcement("14:25", "Strandvik", "Ekö"),
    announcement("14:10", "Ekö", "Strandvik"),
    announcement("14:40", "Ekö", "Strandvik", Info=["Reduced capacity: one vessel in service"]),
    announcement("15:30", "Ekö", "Strandvik"),
    announcement("14:55", "Strandvik", "Ekö", Deleted=True),
)


class TestRequest:
    def test_the_key_and_route_are_escaped_into_the_xml(self):
        root = ElementTree.fromstring(request_body('k"<ey', 'Ek"ö'))
        assert root.find("LOGIN").get("authenticationkey") == 'k"<ey'
        assert root.find(".//EQ[@name='Route.Name']").get("value") == 'Ek"ö'

    def test_the_query_asks_for_departures_still_ahead_in_order(self):
        body = request_body("key", ROUTE)
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
        departures = parse(answer({"DepartureTime": "2026-11-17T14:10:00+01:00"}, announcement("14:20", "Ekö", "Strandvik")))
        assert len(departures) == 1

    def test_a_notice_given_as_one_text_is_one_notice(self):
        """`Info` is a list in the schema; an answer that carries a bare string must not be read letter by letter."""
        departure, = parse(answer(announcement("14:10", "Ekö", "Strandvik", Info="Delayed 10 minutes")))
        assert departure.info == ("Delayed 10 minutes",)

    def test_blank_notices_are_dropped(self):
        departure, = parse(answer(announcement("14:10", "Ekö", "Strandvik", Info=["", "  ", "Ice on the route"])))
        assert departure.info == ("Ice on the route",)


class TestMatching:
    @pytest.mark.parametrize("asked", ["Ekö", "eko", "EKÖ FÄRJELÄGE"])
    def test_a_loosely_named_harbor_matches(self, asked):
        assert matches("Ekö Färjeläge", asked)

    def test_another_harbor_does_not(self):
        assert not matches("Strandvik", "Ekö")


class TestRender:
    def test_both_directions_by_default_with_minutes_for_the_near_ones(self):
        text = render(ROUTE, parse(TIMETABLE), NOW)
        outward = next(line for line in text.splitlines() if line.startswith("From Ekö"))
        back = next(line for line in text.splitlines() if line.startswith("From Strandvik"))
        assert "14:10 (in 8 min)" in outward and "15:30" in outward and "(in 88" not in outward
        assert "14:25 (in 23 min)" in back

    def test_one_side_only_when_an_origin_is_named(self):
        text = render(ROUTE, parse(TIMETABLE), NOW, origin="strandvik")
        assert "From Strandvik" in text and "From Ekö" not in text

    def test_notices_of_shown_departures_are_listed(self):
        assert "Reduced capacity" in render(ROUTE, parse(TIMETABLE), NOW)

    def test_the_count_limits_each_direction(self):
        text = render(ROUTE, parse(TIMETABLE), NOW, origin="Ekö", count=1)
        assert "14:10" in text and "14:40" not in text

    def test_an_unknown_harbor_lists_the_known_ones(self):
        text = render(ROUTE, parse(TIMETABLE), NOW, origin="Göteborg")
        assert "Strandvik" in text and "From" not in text

    def test_a_departure_tomorrow_carries_its_day(self):
        late = answer({**announcement("00:00", "Ekö", "Strandvik"), "DepartureTime": "2026-11-18T06:00:00+01:00"})
        assert "Wed 06:00" in render(ROUTE, parse(late), NOW)


def configured(route):
    return patch.object(module.config, "get", side_effect=lambda key, default=None:
                        {"ferry.api_key": "key", "ferry.route": route}.get(key, default))


class TestRoutes:
    def test_deleted_routes_are_left_out(self):
        assert parse_routes(route_list("Ekeleden", deleted=("Gamlaleden",))) == ["Ekeleden"]

    @pytest.mark.parametrize("asked", ["Älvleden", "älvleden", "ALVLEDEN", " alvleden "])
    def test_a_route_resolves_in_any_case_with_or_without_the_dots(self, asked):
        assert resolve(asked, parse_routes(ROUTES)) == "Älvleden"

    def test_a_part_of_a_name_is_not_a_route(self):
        assert resolve("Älv", parse_routes(ROUTES)) is None

    @pytest.mark.parametrize("asked, meant", [("Älv", "Älvleden"), ("Strömsfärjan", "Strömsleden"),
                                              ("norleden", "Norrleden")])
    def test_the_closest_routes_lead_with_the_meant_one(self, asked, meant):
        assert closest(asked, parse_routes(ROUTES))[0] == meant

    def test_nothing_alike_suggests_nothing(self):
        assert closest("Xyzzy", parse_routes(ROUTES)) == []


class TestFetch:
    @pytest.fixture
    def post(self):
        with configured(ROUTE), patch.object(module.requests, "post") as post:
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
        post.side_effect = api(TIMETABLE)
        FerryIntegration().fetch()
        for call in post.call_args_list:
            assert "key" not in call.args[0].split("/")[-1] and b'authenticationkey="key"' in call.kwargs["data"]

    def test_the_route_list_is_asked_once_for_many_questions(self, post):
        post.side_effect = api(TIMETABLE)
        ferry = FerryIntegration()
        ferry.fetch()
        ferry.fetch()
        assert sum(b"FerryRoute" in c.kwargs["data"] for c in post.call_args_list) == 1

    def test_a_route_withdrawn_since_the_list_was_read_is_unknown_once_the_list_is_old(self, post):
        clock = Mock(monotonic=Mock(return_value=1000.0))
        ferry = FerryIntegration()
        with patch.object(module, "time", clock):
            post.side_effect = api(TIMETABLE)
            ferry.fetch()
            post.side_effect = api(TIMETABLE, route_list("Norrleden", "Bodaleden"))
            clock.monotonic.return_value += module.ROUTES_TTL - 1
            assert ferry.fetch()                      # the list it read still stands
            clock.monotonic.return_value += 2
            with pytest.raises(FerryError, match="Unknown ferry route"):
                ferry.fetch()

    def test_an_error_page_instead_of_json_is_a_ferry_error_naming_the_status(self, post):
        post.return_value = Mock(status_code=502, json=Mock(side_effect=ValueError("no JSON")))
        with pytest.raises(FerryError, match="502"):
            FerryIntegration().fetch()

    def test_a_connection_that_cannot_be_made_is_a_ferry_error(self, post):
        post.side_effect = requests.ConnectionError()
        with pytest.raises(FerryError):
            FerryIntegration().fetch()


def departures_query(post) -> str:
    bodies = [c.kwargs["data"].decode() for c in post.call_args_list if b"FerryAnnouncement" in c.kwargs["data"]]
    return ElementTree.fromstring(bodies[-1]).find(".//EQ[@name='Route.Name']").get("value")


class TestConfiguredRoute:
    def test_the_query_names_the_route_from_config(self):
        with configured("Norrleden"), patch.object(module.requests, "post", side_effect=api(answer())) as post:
            FerryIntegration().fetch()
        assert departures_query(post) == "Norrleden"

    def test_a_loosely_written_route_is_asked_under_trafikverkets_name(self):
        with configured("alvleden"), patch.object(module.requests, "post", side_effect=api(TIMETABLE)) as post:
            text = FerryIntegration().departures(now=NOW)
        assert departures_query(post) == "Älvleden" and text.startswith("Älvleden")

    @pytest.mark.parametrize("route", ["", "  ", None])
    def test_no_route_is_a_ferry_error_and_no_request(self, route):
        with configured(route), patch.object(module.requests, "post") as post:
            with pytest.raises(FerryError, match="ferry.route"):
                FerryIntegration().fetch()
        post.assert_not_called()

    def test_an_unset_route_leaves_the_integration_not_configured(self):
        with patch("common.ConfigManager.config.get", side_effect=lambda key, default=None:
                   {"ferry.api_key": "key", "ferry.route": ""}.get(key, default)):
            assert FerryIntegration().missing_config_keys() == ["ferry.route"]

    def test_an_empty_route_list_lets_the_name_through_as_written(self):
        with configured("Norrleden"), patch.object(module.requests, "post",
                                                   side_effect=api(answer(), routes=route_list())) as post:
            FerryIntegration().fetch()
        assert departures_query(post) == "Norrleden"


class TestHealth:
    def check(self, route, departures=TIMETABLE, routes=ROUTES):
        with configured(route), patch.object(module.requests, "post", side_effect=api(departures, routes)) as post:
            try:
                return FerryIntegration().check_health(), post
            except FerryError as e:
                return e, post

    def test_a_known_route_with_nothing_ahead_is_healthy(self):
        assert self.check("Ekeleden", departures=answer())[0] is True

    def test_an_unknown_route_names_the_closest_and_asks_for_no_departures(self):
        error, post = self.check("Strömsfärjan")
        assert isinstance(error, FerryError)
        assert "Strömsleden?" in str(error) and "ferry.route" in str(error)
        assert not any(b"FerryAnnouncement" in c.kwargs["data"] for c in post.call_args_list)

    def test_a_partial_name_is_unknown_too(self):
        error, _ = self.check("Älv")
        assert isinstance(error, FerryError) and "Älvleden?" in str(error)

    def test_a_short_route_list_is_given_whole(self):
        error, _ = self.check("Xyzzy")
        assert all(name in str(error) for name in parse_routes(ROUTES))

    def test_a_long_route_list_is_not(self):
        many = route_list(*(f"Ö{n}leden" for n in range(module.LIST_ALL_LIMIT + 1)), "Ekeleden")
        error, _ = self.check("Ekelden", routes=many)
        assert "did you mean Ekeleden" in str(error) and "Ö7leden" not in str(error)
