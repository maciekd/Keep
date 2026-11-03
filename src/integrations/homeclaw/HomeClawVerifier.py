"""The check of the HomeKit automations' scheduled runs, in code: did every accessory a run sets
end up as set, and only a deviation wakes the assistant (issue #110).

A watch item used to do this — a model round 3 minutes before each run to snapshot the lamps
and one 2 minutes after to compare — twice per run, five runs a day, almost always "nothing
missed". Here `plan` finds the next run from `HomeClaw.automations()`: every enabled
automation with a time trigger (a clock time, or sunrise / sunset with an offset, which the
home's coordinates and SunTimes resolve) whose weekday condition fits, the automations of one
minute grouped into one Run, and what they set read from their action sets (`get_scene`) as
Expectations — power, brightness, a target position or temperature. An automation with a
condition code cannot evaluate (a sensor, someone at home) is left out: whether it fired is
unknown, so a mismatch would mean nothing.

AutomationVerifier's thread reads the accessories (`get_accessory` with a live refresh, the
Hue and Plejd ones included although the prompt section leaves them out) LEAD_SECONDS before
the run and again `verify_after_seconds` after it — longer than the retry of whatever drives
the lamps (the paced Home Assistant automations retry after 90 s). Every expectation met is a
log line. A deviation becomes one system turn in the owner's chat (`Notifier.open_conversation`,
a system note when no chat is reachable): the accessory, what the run sets, what it reports, the
state before the run and the app's event log for it since then. The assistant decides what to
do with that — the text states facts and asks for nothing."""
import re
import threading
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from assistant.Notifier import notifier
from assistant.SystemNote import system_note
from common.ConfigManager import config
from common.logging import error, info, trace
from .HomeClawAutomations import value_text
from .HomeClawClient import HomeClawError, HomeClawUnavailable
from .SunTimes import sun_times

LEAD_SECONDS = 180              # the snapshot before a run
DEFAULT_SETTLE_SECONDS = 180    # the wait after a run before the check: the lamps' own retry window and a margin
REPLAN_SECONDS = 3600           # the longest the thread waits before reading the automations again
RETRY_SECONDS = 300             # after the app could not be read
HORIZON_DAYS = 2                # how far ahead a run is looked for: a Mon–Fri automation seen on a Saturday
TIMED_TRIGGERS = ("calendar", "significant_time")
CHECKED = ("power", "active", "brightness", "target_position", "target_temperature", "target_heating_cooling")
SWITCHES = ("power", "active")
NUMERIC_TOLERANCE = 1.0         # a lamp set to 30 % that reports 29 did take the change
EVENT_LIMIT = 500               # events per read of the app's log, newest first
_SUN = re.compile(r"\bat (sunrise|sunset)\s*([+-]\s*\d+)?", re.I)   # a sun trigger's event_summary: "at sunset-45m"
_CLOCK = re.compile(r"^(\d{1,2}):(\d{2})$")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_ON = {"1", "true", "on", "yes"}
_OFF = {"0", "false", "off", "no"}
_HEATING = {"0": "off", "1": "heat", "2": "cool", "3": "auto"}
SOURCE = "HomeClaw"


@dataclass(frozen=True)
class Expectation:
    accessory_id: str
    accessory: str
    characteristic: str
    value: str                  # normalised: "on" / "off", a number, a heating mode
    automation: str


@dataclass(frozen=True)
class Deviation:
    expectation: Expectation
    actual: Optional[str]       # None: the accessory could not be read

    @property
    def accessory_id(self) -> str:
        return self.expectation.accessory_id


@dataclass
class Run:
    at: datetime                # local, aware
    automations: list[str]
    expectations: list[Expectation]

    @property
    def accessory_ids(self) -> list[str]:
        return list(dict.fromkeys(e.accessory_id for e in self.expectations))

    @property
    def names(self) -> str:
        return ", ".join(self.automations)


def normalise(characteristic: str, value) -> str:
    """A characteristic value as both a scene action and an accessory reading spell it:
    a switch is on / off (the app writes 1 / 0 for an action and true / false for a reading),
    a number is its bare digits ("21.0°C" → "21"), a heating mode its name."""
    text = str(value if value is not None else "").strip()
    low = text.lower()
    if characteristic in SWITCHES:
        return "on" if low in _ON else "off" if low in _OFF else low
    if characteristic == "target_heating_cooling":
        return _HEATING.get(text, low)
    number = _NUMBER.search(text)
    return f"{float(number.group()):g}" if number else low


def same(characteristic: str, expected: str, actual: str) -> bool:
    try:
        return abs(float(expected) - float(actual)) <= NUMERIC_TOLERANCE
    except ValueError:
        return expected == actual


def applies(automation: dict, day: date) -> Optional[bool]:
    """Whether the automation fires on `day`: False when it is disabled or a weekdays condition
    leaves the day out, None when a condition of another kind decides (a sensor, who is home) —
    code cannot tell whether it fired, so it is not verified."""
    if not automation.get("enabled", True):
        return False
    verdict = True
    for condition in automation.get("conditions") or []:
        if condition.get("type") == "weekdays":
            apple_weekday = day.isoweekday() % 7 + 1       # HomeKit counts Sunday as 1
            if apple_weekday not in {int(d) for d in condition.get("weekdays") or []}:
                return False
        else:
            verdict = None
    return verdict


def fire_at(automation: dict, day: date, tz: ZoneInfo,
            sun: Callable[[date], Optional[tuple[datetime, datetime]]]) -> Optional[datetime]:
    """When the automation fires on `day`: a calendar trigger's clock time, a sun trigger's
    event with its offset through `sun` (None when the sun's times are unknown or the sun
    neither rises nor sets that day); None for every other trigger."""
    kind = automation.get("trigger_type")
    if kind == "calendar":
        match = _CLOCK.match(str(automation.get("fire_time") or ""))
        return datetime.combine(day, time(int(match[1]), int(match[2])), tzinfo=tz) if match else None
    if kind == "significant_time":
        match = _SUN.search(str(automation.get("event_summary") or ""))
        times = sun(day) if match else None
        if times is None:
            return None
        base = times[0] if match[1].lower() == "sunrise" else times[1]
        return base + timedelta(minutes=int(match[2].replace(" ", "")) if match[2] else 0)
    return None


def expectations_of(automation: dict, actions_of: Callable[[str], list[dict]]) -> list[Expectation]:
    """What the automation sets that the check can read back: the CHECKED characteristics of
    every action of its action sets (`actions_of` is `get_scene`'s actions for an id)."""
    name = str(automation.get("name") or automation.get("id"))
    found = []
    for action_set in automation.get("action_sets") or []:
        for action in actions_of(str(action_set.get("id") or "")):
            characteristic = str(action.get("characteristic") or "")
            uid = str(action.get("accessory_id") or "").upper()
            if characteristic in CHECKED and uid:
                found.append(Expectation(uid, str(action.get("accessory") or uid), characteristic,
                                         normalise(characteristic, action.get("value")), name))
    return found


def plan(automations: list[dict], now: datetime, sun: Callable[[date], Optional[tuple[datetime, datetime]]],
         actions_of: Callable[[str], list[dict]], after: Optional[datetime] = None) -> Optional[Run]:
    """The next run after `now` within HORIZON_DAYS: the earliest minute a verifiable timed
    automation fires, every automation of that minute in it, with what they set. A minute
    whose automations set nothing the check can read is passed over. With `after` — the
    minute following the last verified run — the runs from that minute on count even when
    they have fired: one that fired while the previous run was being checked (the wait
    outlasts a run a few minutes later) is verified late rather than skipped."""
    tz = now.tzinfo
    start = now if after is None or after > now else after
    moments: dict[datetime, list[dict]] = {}
    for ahead in range(HORIZON_DAYS + 1):
        day = (start + timedelta(days=ahead)).date()
        for automation in automations:
            if automation.get("trigger_type") not in TIMED_TRIGGERS:
                continue
            at = fire_at(automation, day, tz, sun)
            if at is None or (at <= now if after is None else at.replace(second=0, microsecond=0) < after):
                continue
            verdict = applies(automation, day)
            if verdict is None:
                trace(f"HomeClaw verifier: {automation.get('name')} has a condition code cannot evaluate — not verified")
            elif verdict:
                moments.setdefault(at.replace(second=0, microsecond=0), []).append(automation)
    for at in sorted(moments):
        group = moments[at]
        expectations = [e for automation in group for e in expectations_of(automation, actions_of)]
        names = [str(a.get("name") or a.get("id")) for a in group]
        if expectations:
            return Run(at, names, expectations)
        trace(f"HomeClaw verifier: {', '.join(names)} at {at:%Y-%m-%d %H:%M} sets nothing the check can read — passed over")
    return None


def read_state(detail: dict) -> dict[str, str]:
    """The CHECKED characteristics of a `get_accessory` answer, normalised; the first service
    that carries each wins (a lamp's lightbulb service comes before its extras)."""
    found: dict[str, str] = {}
    for service in detail.get("services") or []:
        for characteristic in service.get("characteristics") or []:
            name = str(characteristic.get("name") or "")
            if name in CHECKED and name not in found:
                found[name] = normalise(name, characteristic.get("value"))
    for name, value in (detail.get("state") or {}).items():
        if name in CHECKED and name not in found:
            found[name] = normalise(name, value)
    return found


def compare(expectations: list[Expectation], states: dict[str, Optional[dict[str, str]]]) -> list[Deviation]:
    """The expectations the readings do not meet. An accessory switched off by the run is held
    to that alone — its brightness then says nothing. One that could not be read (None in
    `states`) did not confirm anything, so each of its expectations is a deviation."""
    switched_off = {e.accessory_id for e in expectations if e.characteristic in SWITCHES and e.value == "off"}
    deviations = []
    for expectation in expectations:
        if expectation.accessory_id in switched_off and expectation.characteristic not in SWITCHES:
            continue
        state = states.get(expectation.accessory_id)
        actual = state.get(expectation.characteristic) if state else None
        if actual is None or not same(expectation.characteristic, expectation.value, actual):
            deviations.append(Deviation(expectation, actual))
    return deviations


def _shown(characteristic: str, value: Optional[str]) -> str:
    if value is None:
        return "unread"
    return value if characteristic in SWITCHES else value_text(characteristic, value) or value


def _settings(expectations: list[Expectation], readings: Optional[dict[str, str]]) -> str:
    """The expectations' characteristics as read: "on, 30 %"."""
    return ", ".join(_shown(e.characteristic, readings.get(e.characteristic) if readings else None) for e in expectations)


def deviation_report(run: Run, before: dict[str, Optional[dict[str, str]]], deviations: list[Deviation],
                     changes: dict[str, list[str]], settle_seconds: int) -> str:
    """The system turn the assistant gets: what the run set, what deviates and how, the state
    before the run, the app's log of the deviating accessories since the snapshot, and how
    many accessories did as set. Facts, no instruction."""
    by_accessory: dict[str, list[Deviation]] = {}
    for deviation in deviations:
        by_accessory.setdefault(deviation.accessory_id, []).append(deviation)
    lines = [f"HomeKit automation check ({SOURCE}): {run.names} fired at {run.at:%H:%M}. "
             f"{settle_seconds // 60} min later these accessories do not report what it sets:"]
    for uid, found in by_accessory.items():
        expected = [d.expectation for d in found]
        actual = {d.expectation.characteristic: d.actual for d in found}
        line = (f"- {found[0].expectation.accessory}: set to {_settings(expected, {e.characteristic: e.value for e in expected})}"
                f", reports {_settings(expected, actual)} (before the run: {_settings(expected, before.get(uid))})")
        if changes.get(uid):
            line += f"; {SOURCE}'s log since {(run.at - timedelta(seconds=LEAD_SECONDS)):%H:%M}: " + ", ".join(changes[uid])
        lines.append(line)
    others = len(run.accessory_ids) - len(by_accessory)
    lines.append("The run sets no other accessory." if not others else "The other accessory reports what the run sets."
                 if others == 1 else f"The other {others} accessories report what the run sets.")
    return "\n".join(lines)


class AutomationVerifier:
    """The thread: plans the next run, reads its accessories LEAD_SECONDS before and
    `settle_seconds` after it, and wakes the assistant on a deviation. `integration` is the
    HomeClawIntegration whose client and automations it reads."""

    def __init__(self, integration, settle_seconds: int = DEFAULT_SETTLE_SECONDS):
        self._integration = integration
        self.settle_seconds = settle_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tz = ZoneInfo(config.get_home_timezone())
        self._no_sun_logged = False
        self.last_run: Optional[Run] = None
        self._verified_until: Optional[datetime] = None     # the minute after the last verified run

    def start(self):
        self._thread = threading.Thread(target=self._loop, name="HomeClawVerifier", daemon=True)
        self._thread.start()
        info(f"HomeClaw verifier: watching the timed automations (check {self.settle_seconds} s after each run)")

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self):
        while not self._stop.is_set():
            try:
                wait = self._cycle()
            except HomeClawUnavailable as e:
                info(f"HomeClaw verifier: {e} — trying again in {RETRY_SECONDS // 60} min")
                wait = RETRY_SECONDS
            except Exception as e:
                error(f"HomeClaw verifier: {type(e).__name__}: {e} — trying again in {RETRY_SECONDS // 60} min")
                wait = RETRY_SECONDS
            self._stop.wait(wait)

    def _cycle(self) -> float:
        """One turn of the loop: the seconds to wait before the next, having verified a run
        whose snapshot moment has come."""
        now = self._now()
        run = self.next_run(now, after=self._verified_until)
        if run is None:
            info("HomeClaw verifier: no timed automation ahead to verify — looking again in an hour")
            return REPLAN_SECONDS
        until_snapshot = (run.at - now).total_seconds() - LEAD_SECONDS
        if until_snapshot > 0:
            trace(f"HomeClaw verifier: next run {run.names} at {run.at:%Y-%m-%d %H:%M}, "
                  f"snapshot in {until_snapshot / 60:.0f} min")
            return min(until_snapshot, REPLAN_SECONDS)
        if run.at <= now:
            info(f"HomeClaw verifier: {run.names} fired at {run.at:%H:%M} while the previous run was being "
                 "checked — verified late, without a reading before it")
        self.verify(run)
        return 1

    def _now(self) -> datetime:
        return datetime.now(self._tz)

    def next_run(self, now: Optional[datetime] = None, after: Optional[datetime] = None) -> Optional[Run]:
        return plan(self._integration.automations(), now or self._now(), self._sun, self._actions_of, after)

    def _sun(self, day: date) -> Optional[tuple[datetime, datetime]]:
        latitude, longitude = config.get_home_latitude(), config.get_home_longitude()
        if latitude is None or longitude is None:
            if not self._no_sun_logged:
                info("HomeClaw verifier: no location.home coordinates — the sunrise/sunset automations are not verified")
                self._no_sun_logged = True
            return None
        return sun_times(day, latitude, longitude, str(self._tz))

    def _actions_of(self, action_set_id: str) -> list[dict]:
        try:
            detail = self._integration.client().call("get_scene", id=action_set_id) or {}
        except HomeClawUnavailable:
            raise
        except HomeClawError as e:
            info(f"HomeClaw verifier: cannot read the actions of {action_set_id}: {e}")
            return []
        return list(detail.get("actions") or [])

    def _read(self, accessory_ids: list[str]) -> dict[str, Optional[dict[str, str]]]:
        """A live reading of each accessory; None for one the app cannot read."""
        states: dict[str, Optional[dict[str, str]]] = {}
        for uid in accessory_ids:
            try:
                states[uid] = read_state(self._integration.client().call("get_accessory", id=uid, refresh=True) or {})
            except HomeClawUnavailable:
                raise
            except HomeClawError as e:
                info(f"HomeClaw verifier: cannot read {uid}: {e}")
                states[uid] = None
        return states

    def _changes(self, accessory_ids: set[str], since: datetime) -> dict[str, list[str]]:
        """The app's event log for the accessories since `since`, a line per change."""
        stamp = since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")      # the one form the app filters on
        try:
            data = self._integration.client().call("events", limit=EVENT_LIMIT, since=stamp, type="characteristic_change")
        except HomeClawError as e:
            info(f"HomeClaw verifier: the event log could not be read: {e}")
            return {}
        events = data.get("events") if isinstance(data, dict) else data
        changes: dict[str, list[str]] = {}
        for event in sorted(events or [], key=lambda e: str(e.get("timestamp") or "")):
            uid = str((event.get("accessory") or {}).get("id") or "").upper()
            if uid not in accessory_ids:
                continue
            try:
                when = datetime.fromisoformat(str(event.get("timestamp")).replace("Z", "+00:00")).astimezone(self._tz)
            except ValueError:
                continue
            changes.setdefault(uid, []).append(
                f"{when:%H:%M:%S} {str(event.get('characteristic') or '?').replace('_', ' ')} "
                f"{event.get('previous_value', '?')} → {event.get('value', '?')}")
        return changes

    def verify(self, run: Run) -> list[Deviation]:
        """The check of one run: the snapshot now, the reading `settle_seconds` after the run,
        the comparison, and the assistant woken when something deviates. Empty when the
        thread was stopped while waiting."""
        ids = run.accessory_ids
        # A run found after it fired has no "before" reading: one taken now would show the
        # run's own effect as the state before it.
        fired = run.at <= self._now()
        before = {} if fired else self._read(ids)
        info(f"HomeClaw verifier: {run.names} {'fired' if fired else 'fires'} at {run.at:%H:%M} — {len(ids)} "
             f"accessories {'to check' if fired else 'read before it'}, checking {self.settle_seconds} s after")
        wait = (run.at - self._now()).total_seconds() + self.settle_seconds
        if self._stop.wait(max(wait, 0)):
            trace(f"HomeClaw verifier: stopped while waiting for the run at {run.at:%H:%M}")
            return []
        after = self._read(ids)
        deviations = compare(run.expectations, after)
        self.last_run = run
        self._verified_until = run.at.replace(second=0, microsecond=0) + timedelta(minutes=1)
        if not deviations:
            info(f"HomeClaw verifier: {run.names} at {run.at:%H:%M} — all {len(ids)} accessories report what the run "
                 "sets, nothing to do")
            return []
        changes = self._changes({d.accessory_id for d in deviations}, run.at - timedelta(seconds=LEAD_SECONDS))
        text = deviation_report(run, before, deviations, changes, self.settle_seconds)
        info(f"HomeClaw verifier: {len(deviations)} deviation(s) after {run.names} at {run.at:%H:%M} — "
             "waking the assistant")
        if not notifier().open_conversation(text):
            info("HomeClaw verifier: no owner chat to open — the report goes to the history as a system note")
            system_note(SOURCE, text)
        return deviations
