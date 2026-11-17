"""The next departures of a Trafikverket car ferry (Färjerederiet), both directions, from
Trafikverket's open data API (https://data.trafikverket.se/).

One POST to `/v2/data.json` per question: a `FerryAnnouncement` query for the route the user
configures (`ferry.route`, required — no route is built in) whose departure is still ahead, ordered
by departure time; both directions come from the harbors the answer names, none is hard-coded.
Each announcement carries `DepartureTime` (Swedish local time with its offset), `FromHarbor` and
`ToHarbor` (`{"Name": ...}`), `Deleted` and an optional `Info` (the deviation notices). The API key
(`ferry.api_key`) goes in the request body's `LOGIN` element, never in a URL or log line."""
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional
from xml.sax.saxutils import quoteattr

import requests

from assistant.integration.Integration import Integration
from common.ConfigManager import config
from common.logging import info, trace

API_URL = "https://api.trafikinfo.trafikverket.se/v2/data.json"
SCHEMA_VERSION = "1.2"
TIMEOUT = 15
# Enough announcements for a busy route's next hour or two in both directions.
QUERY_LIMIT = 60
DEFAULT_COUNT = 4
_FOLD = str.maketrans("åäöéü", "aaoeu")


class FerryError(Exception):
    """The departures could not be fetched; the text is for the LLM."""


@dataclass(frozen=True)
class Departure:
    time: datetime
    origin: str
    destination: str
    info: tuple[str, ...] = ()


def fold(name: str) -> str:
    return " ".join(name.lower().translate(_FOLD).split())


def matches(harbor: str, asked: str) -> bool:
    """A harbor the user named loosely: part of its name, any case, with or without å/ä/ö."""
    harbor, asked = fold(harbor), fold(asked)
    return bool(asked) and (asked in harbor or harbor in asked)


def request_body(api_key: str, route: str, limit: int = QUERY_LIMIT) -> str:
    return (f"<REQUEST><LOGIN authenticationkey={quoteattr(api_key)}/>"
            f"<QUERY objecttype=\"FerryAnnouncement\" schemaversion=\"{SCHEMA_VERSION}\" "
            f"orderby=\"DepartureTime\" limit=\"{int(limit)}\">"
            f"<FILTER><AND><EQ name=\"Route.Name\" value={quoteattr(route)}/>"
            f"<GT name=\"DepartureTime\" value=\"$now\"/>"
            f"<EQ name=\"Deleted\" value=\"false\"/></AND></FILTER></QUERY></REQUEST>")


def parse(payload: dict) -> List[Departure]:
    """The departures in a `data.json` answer, sorted; raises FerryError for the API's own error."""
    try:
        result = payload["RESPONSE"]["RESULT"][0]
    except (KeyError, IndexError, TypeError):
        raise FerryError("Trafikverket answered with something that is not a ferry result")
    if "ERROR" in result:
        raise FerryError(f"Trafikverket refused the query: {result['ERROR'].get('MESSAGE', 'no reason given')}")
    departures = []
    for item in result.get("FerryAnnouncement", []):
        if item.get("Deleted"):
            continue
        try:
            time = datetime.fromisoformat(item["DepartureTime"])
            origin, destination = item["FromHarbor"]["Name"], item["ToHarbor"]["Name"]
        except (KeyError, TypeError, ValueError):
            trace(f"Ferry: announcement {item.get('Id')} has no readable time or harbors, skipped")
            continue
        notes = item.get("Info") or ()
        notes = (notes,) if isinstance(notes, str) else tuple(str(n) for n in notes)
        departures.append(Departure(time, origin, destination, tuple(n for n in notes if n.strip())))
    return sorted(departures, key=lambda d: d.time)


def _clock(time: datetime, now: datetime) -> str:
    clock = time.strftime("%H:%M")
    if time.date() != now.astimezone(time.tzinfo).date():
        clock = f"{time.strftime('%a')} {clock}"
    minutes = int((time - now).total_seconds() // 60)
    return f"{clock} (in {minutes} min)" if minutes < 60 else clock


def render(route: str, departures: Iterable[Departure], now: datetime, origin: str = "",
           count: int = DEFAULT_COUNT) -> str:
    """The next `count` departures per direction, or only from the harbor `origin` names."""
    departures = [d for d in departures if d.time > now]
    harbors = sorted({d.origin for d in departures})
    if origin:
        harbors = [h for h in harbors if matches(h, origin)]
        if not harbors:
            known = ", ".join(sorted({d.origin for d in departures})) or "none"
            return f"No departures from a harbor matching {origin!r} on {route}. Its harbors: {known}"
    if not harbors:
        return f"No upcoming departures found for {route}."
    lines = [f"{route}, next departures (now {now.astimezone(departures[0].time.tzinfo).strftime('%H:%M')}):"]
    notes: list[str] = []
    for harbor in harbors:
        ahead = [d for d in departures if d.origin == harbor][:count]
        destinations = " / ".join(dict.fromkeys(d.destination for d in ahead))
        lines.append(f"From {harbor} to {destinations}: " + ", ".join(_clock(d.time, now) for d in ahead))
        for d in ahead:
            notes.extend(n for n in d.info if n not in notes)
    if notes:
        lines.append("Notices: " + " | ".join(notes))
    return "\n".join(lines)


class FerryIntegration(Integration):
    """The next departures of the user's chosen Trafikverket car ferry route.

    Answers when the ferry leaves next from either side of its route, with Trafikverket's
    notices about delays and changes.

    Reads Trafikverket's open data API (`FerryAnnouncement`) with the household's API key on
    every question; nothing is cached or polled."""

    @property
    def name(self) -> str:
        return "Ferry"

    @property
    def config_keys(self) -> List[str]:
        return ["ferry.api_key", "ferry.route"]

    @staticmethod
    def route() -> str:
        return str(config.get("ferry.route", "") or "").strip()

    def fetch(self, limit: int = QUERY_LIMIT) -> List[Departure]:
        route = self.route()
        if not route:
            raise FerryError("No ferry route is configured: ferry.route in config.yaml names it, "
                             "as Trafikverket writes it")
        body = request_body(config.get("ferry.api_key"), route, limit)
        try:
            response = requests.post(API_URL, data=body.encode("utf-8"),
                                     headers={"Content-Type": "text/xml; charset=utf-8"}, timeout=TIMEOUT)
        except requests.Timeout:
            raise FerryError("Trafikverket's API did not answer in time")
        except requests.ConnectionError:
            raise FerryError("Could not connect to Trafikverket's API")
        if response.status_code in (401, 403):
            raise FerryError("Trafikverket refused the API key (ferry.api_key in config.yaml)")
        try:
            payload = response.json()
        except ValueError:
            raise FerryError(f"Trafikverket answered HTTP {response.status_code} with something that is not JSON")
        departures = parse(payload)
        trace(f"Ferry: {len(departures)} upcoming departures on {route}")
        return departures

    def check_health(self) -> bool:
        if not self.fetch(limit=1):
            raise FerryError(f"Trafikverket lists no upcoming departures for the route {self.route()!r} "
                             f"— is ferry.route spelled as Trafikverket names it?")
        return True

    def departures(self, origin: str = "", now: Optional[datetime] = None) -> str:
        route = self.route()
        info(f"Ferry: next departures on {route}" + (f" from {origin!r}" if origin else ""))
        now = now or datetime.now().astimezone()
        return render(route, self.fetch(), now, origin, int(config.get("ferry.count", DEFAULT_COUNT)))
