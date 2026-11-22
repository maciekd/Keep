"""Writes to an owner who has gone quiet: when a message is due, what the periodic check is told then, and what the message leaves behind.

Off unless `reactivation.enabled` is true. The clock is the owner's last turn — `owner_spoke`,
called by `Assistant.stream` for every request that is neither a guest's nor one the server
started — and runs whether the key is on or not, in `cache/reactivation.json`, so a restart does
not start it over and switching the key on finds a clock that is true. An install without the
file (one that ran before this code, a restored one) gets its clock from the history when the key
is on: the last row somebody wrote to the assistant (`_seed`). A guest's row counts there, which
can only make the silence shorter than it was, and a history with no such row leaves the install
without a clock until the owner's next turn: nothing here claims a silence it did not measure.
The file goes with a `//wipe` and is in no backup.

The schedule is counted from that turn and each step is due `LEAD_SECONDS` ahead of it, so a
message is there a little before the time of day the owner was last active (the periodic check
notices a step up to its own interval late): one a day for `DAILY_STEPS` days, the last of them
saying that the daily ones stop — when one of them went out: for an owner nobody has written to
in this silence it is a message like the others —, then one every `FORTNIGHT_DAYS`. The periodic
check asks `due()` and carries the step's `note()`; the model writes the message and picks where
it goes, among the `post_message` destinations. `record()` then spends the step — sent, or
declined when the model answered NO — so it is not asked again; a check the model gave no
answer to (a refusal) leaves it due. One message per check and none within `MIN_GAP_SECONDS` of
the last: the steps that passed while the server was down are one message, the newest that is
due — and it waits for its time of day: a step found more than `SLOT_SECONDS` after an hour it
was due at (the server was down, the key or the clock is new) is held until that hour comes round
again, so no message goes out at whatever hour a restart happened to be. The one late step that
goes out at once is the one the house's Night mode held back (`after_night`): the morning is when
the owner is there. A turn of the owner's starts the schedule over.

The silence has two sides. A step goes out only when the assistant has been quiet too: a message
it chose to send to a chat in the last `COVER_SECONDS` — a `post_message` of any check or turn,
noted by `assistant_wrote`; the notices the code writes (a restart, an update) are no messages —
is the day's contact with the owner, and the step is spent as covered. A message from before the
owner's last turn covers nothing: the owner has answered since, and the silence began after it.
A step's own message covers nothing either: the schedule's next step is the one after it.

Telemetry: `reactivation_nudge` for every step spent (covered ones too), `reactivation_return` for the owner's
first turn after a silence of `RETURN_AFTER_SECONDS` or more — with no message sent meanwhile
too, which is the base rate the returns after a message are read against.
`docs/plans/REACTIVATION_PLAN.md` has the reasoning."""
import json
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from assistant import Destinations, WatchList
from assistant.ChannelProfile import SOURCE_AUTONOMOUS
from assistant.Notifier import notifier
from assistant.Telemetry import Telemetry
from common.ConfigManager import config
from common.FloxyDatabase import FloxyDatabase
from common.Paths import CACHE_DIR
from common.logging import error, info, trace

STATE_FILE = CACHE_DIR / "reactivation.json"
ENABLED_KEY = "reactivation.enabled"

DAY = 24 * 3600
# A step is due this long before the time of day of the owner's last turn. The periodic check
# notices it 0–55 minutes later, so the message lands half an hour early on average and never late.
LEAD_SECONDS = 3600
# How long after its hour a step still goes out: two checks' worth. Later than that it waits
# for the same hour of the next day.
SLOT_SECONDS = 2 * 3600
DAILY_STEPS = 7             # one message a day for this many days; the last says the daily ones stop
FORTNIGHT_DAYS = 14         # then one every this many days
MIN_GAP_SECONDS = 12 * 3600     # between two steps, whatever fell due meanwhile
RETURN_AFTER_SECONDS = DAY      # a turn after a silence this long is a return
COVER_SECONDS = DAY             # a message of the assistant's own this recent stands in for the step

KIND_DAILY = "daily"
KIND_BACKOFF = "backoff"
KIND_FORTNIGHTLY = "fortnightly"
OUTCOME_SENT = "sent"
OUTCOME_DECLINED = "declined"
OUTCOME_COVERED = "covered"     # never asked of the model: the assistant had written already
UNKNOWN_DESTINATION = "unknown"

NOTE = """
## QUIET OWNER

{name} last wrote {quiet} ago.{backoff}

Send them one message with `post_message` that gives them a reason to come back. One destination, one message. If the history or the calendar says why they are away and a message would be unwelcome now, or nothing listed would reach them, answer NO."""

# The one thing of the schedule the model cannot work out for itself.
BACKOFF = f" This is the last daily message, after it you write once every {FORTNIGHT_DAYS} days: say so in it."

_lock = threading.Lock()


@dataclass(frozen=True)
class Step:
    """One message of the schedule, as `due()` found it."""
    number: int             # 1 … DAILY_STEPS are the daily ones, the fortnightly ones count on from there
    since: float            # the owner's last turn the schedule is counted from
    quiet_seconds: float    # how long the owner had been quiet when the step was found due
    sent_before: int        # the messages that went out earlier in this silence

    @property
    def kind(self) -> str:
        if self.number > DAILY_STEPS:
            return KIND_FORTNIGHTLY
        # The last daily one calls the daily ones off, and there is nothing to call off for an
        # owner who got none: the key switched on a week into a silence, a server that was
        # down, six steps the model declined.
        return KIND_BACKOFF if self.number == DAILY_STEPS and self.sent_before else KIND_DAILY

    @property
    def position(self) -> str:
        """Where the step stands in the schedule, as the history row says it."""
        if self.kind == KIND_DAILY:
            return f"message {self.number} of {DAILY_STEPS} daily ones"
        if self.kind == KIND_BACKOFF:
            return f"the last of the {DAILY_STEPS} daily messages, after it one every {FORTNIGHT_DAYS} days"
        return f"one of the messages sent every {FORTNIGHT_DAYS} days, the daily ones are over"


def due_at(last_turn: float, number: int) -> float:
    """When step *number* of a silence that began at *last_turn* is due."""
    days = number if number <= DAILY_STEPS else DAILY_STEPS + FORTNIGHT_DAYS * (number - DAILY_STEPS)
    return last_turn + days * DAY - LEAD_SECONDS


def newest_step(last_turn: float, now: float) -> int:
    """The number of the newest step whose time has come, 0 before the first."""
    days = (now - last_turn + LEAD_SECONDS) / DAY
    if days < DAILY_STEPS + FORTNIGHT_DAYS:
        return min(int(days), DAILY_STEPS)
    return DAILY_STEPS + int((days - DAILY_STEPS) // FORTNIGHT_DAYS)


def _describe(seconds: float) -> str:
    return WatchList.describe_duration(timedelta(seconds=seconds))


def _time(value) -> Optional[float]:
    return None if value is None else float(value)


def _load() -> Optional[dict]:
    try:
        with open(STATE_FILE) as f:
            raw = json.load(f)
        # Either time can be missing: a file the assistant's message made before any turn of
        # the owner's, a file from before the messages were noted.
        return {"last_owner_turn": _time(raw.get("last_owner_turn")), "last_message": _time(raw.get("last_message")),
                "spent": [{"step": int(s["step"]), "at": float(s["at"]), "outcome": str(s["outcome"])}
                          for s in raw.get("spent", [])]}
    except FileNotFoundError:
        return None
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        error(f"Reactivation: the state file cannot be read ({type(e).__name__}: {e}) — "
              f"the clock starts over at the owner's next turn")
        return None


def _save(state: dict) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    except OSError as e:
        error(f"Reactivation: the state file cannot be written: {e}")


def _seed(last_message: Optional[float]) -> Optional[dict]:
    """The clock of an install that has none, read back out of the history and kept: the last
    row somebody wrote to the assistant. None when the history holds no such row."""
    last_turn = FloxyDatabase.get_instance().last_dialog_turn(skip_source=SOURCE_AUTONOMOUS)
    if last_turn is None:
        trace("Reactivation: no turn of the owner's on record and none in the history, nothing to count a silence from")
        return None
    state = {"last_owner_turn": last_turn, "last_message": last_message, "spent": []}
    _save(state)
    info(f"Reactivation: no clock on record — started from the last turn in the history, "
         f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(last_turn))}")
    return state


def owner_spoke(source: Optional[str], now: Optional[float] = None) -> None:
    """The owner said something: the silence ends, the schedule starts over from this turn, and
    a turn that ends a silence of `RETURN_AFTER_SECONDS` or more is reported as a return. Called
    on the request's own thread, so nothing in here may fail the request."""
    try:
        now = time.time() if now is None else now
        with _lock:
            state = _load()
            _save({"last_owner_turn": now, "last_message": state and state["last_message"], "spent": []})
        if state is None or state["last_owner_turn"] is None:
            trace("Reactivation: the owner's first turn on record — the clock starts here")
            return
        quiet = now - state["last_owner_turn"]
        if quiet < RETURN_AFTER_SECONDS:
            return
        sent = [s for s in state["spent"] if s["outcome"] == OUTCOME_SENT]
        last = max(sent, key=lambda s: s["at"], default=None)
        info(f"Reactivation: the owner is back after {_describe(quiet)}, by {source or 'an unknown way in'} — "
             f"{len(sent)} message(s) were sent meanwhile")
        Telemetry.get_instance().reactivation_return(
            inactive_hours=quiet / 3600, nudges_sent=len(sent), last_step=last["step"] if last else None,
            hours_since_nudge=(now - last["at"]) / 3600 if last else None, source=source)
    except Exception as e:
        error(f"Reactivation: noting the owner's turn failed: {type(e).__name__}: {e}")


def assistant_wrote(now: Optional[float] = None) -> None:
    """The assistant sent a message of its own choosing to a chat: for `COVER_SECONDS` no step
    goes out on top of it. Noted whether the key is on or not, like the owner's turn, and on
    the thread of the command that sent it, so nothing in here may fail the command."""
    try:
        now = time.time() if now is None else now
        with _lock:
            state = _load() or {"last_owner_turn": None, "spent": []}
            _save({**state, "last_message": now})
        trace("Reactivation: the assistant wrote to a chat — no step goes out on top of it")
    except Exception as e:
        error(f"Reactivation: noting the assistant's message failed: {type(e).__name__}: {e}")


def due(now: Optional[float] = None, after_night: bool = False) -> Optional[Step]:
    """The step the periodic check should carry now, None when there is none: the key is off,
    the owner has no turn on record, no step's time has come, the newest one is spent, the
    last one was spent less than `MIN_GAP_SECONDS` ago, the step is past its hour and waits
    for the next — unless *after_night* says this is the first check since Night mode ended —
    or the assistant wrote to a chat within `COVER_SECONDS`, which spends the step as covered."""
    if not config.get(ENABLED_KEY, False):
        trace(f"Reactivation: off ({ENABLED_KEY} is not true)")
        return None
    now = time.time() if now is None else now
    with _lock:
        state = _load()
        if state is None or state["last_owner_turn"] is None:
            state = _seed(state and state["last_message"])
    if state is None:
        return None
    last_turn, spent = state["last_owner_turn"], state["spent"]
    quiet = now - last_turn
    number = newest_step(last_turn, now)
    if number == 0:
        trace(f"Reactivation: the owner wrote {_describe(quiet)} ago — the first message is due at "
              f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(due_at(last_turn, 1)))}")
        return None
    if any(s["step"] >= number for s in spent):
        trace(f"Reactivation: quiet for {_describe(quiet)}, step {number} is spent — the next is due at "
              f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(due_at(last_turn, number + 1)))}")
        return None
    last_spent = max((s["at"] for s in spent), default=None)
    if last_spent is not None and now - last_spent < MIN_GAP_SECONDS:
        trace(f"Reactivation: step {number} is due, but the last one was spent {_describe(now - last_spent)} ago "
              f"— it waits until {MIN_GAP_SECONDS // 3600} h have passed")
        return None
    past_its_hour = (now - last_turn + LEAD_SECONDS) % DAY
    if past_its_hour > SLOT_SECONDS and not after_night:
        trace(f"Reactivation: step {number} is due, but {_describe(past_its_hour)} past its hour — it waits until "
              f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(now + DAY - past_its_hour))}")
        return None
    step = Step(number=number, since=last_turn, quiet_seconds=quiet,
                sent_before=sum(s["outcome"] == OUTCOME_SENT for s in spent))
    # The message a step sent is the schedule's own and covers nothing, and neither does one
    # the owner has answered since: the silence is counted from that answer.
    last_message = state["last_message"]
    last_sent = max((s["at"] for s in spent if s["outcome"] == OUTCOME_SENT), default=0.0)
    if last_message is not None and last_message > max(last_sent, last_turn) and now - last_message < COVER_SECONDS:
        info(f"Reactivation: step {number} is due, but the assistant wrote to a chat {_describe(now - last_message)} ago "
             f"— that is the contact, the step is spent as covered")
        _spend(step, OUTCOME_COVERED, [], now)
        return None
    info(f"Reactivation: the owner has been quiet for {_describe(quiet)} — step {number} is due ({step.position})")
    return step


def _channels() -> list[tuple[str, bool]]:
    """What `owner` reaches now, for the telemetry: each chat a proactive push goes to, and
    whether somebody can see it at the moment."""
    try:
        return list(notifier().owner_channels())
    except Exception as e:
        error(f"Reactivation: the notifier could not say what the owner's chats are: {type(e).__name__}: {e}")
        return []


def note(step: Step) -> str:
    """What the check that carries *step* is told, as a turn after its instruction: the fact
    and the ask. What to write and where it goes are the model's to work out."""
    return NOTE.format(name=config.get("owner_name") or "The owner", quiet=_describe(step.quiet_seconds),
                       backoff=BACKOFF if step.kind == KIND_BACKOFF else "")


def opening(step: Step) -> str:
    """What the check was, for the row that opens its rows in the history: a later chat turn
    then knows what its own message was, and what to set when the owner wants no more of them."""
    return (f"the owner has been quiet for {_describe(step.quiet_seconds)}: {step.position} — "
            f"the config key {ENABLED_KEY} turns these messages off")


def posted_kinds(executed_commands: list[str]) -> list[str]:
    """The kind of the destination of every `post_message` among *executed_commands* (the
    processor's `name(params)` record of the calls that succeeded)."""
    kinds = []
    for call in executed_commands:
        name, _, params = call.partition("(")
        if name.strip() != "post_message":
            continue
        kinds.append(_posted_kind(params))
    return kinds


def _posted_kind(params: str) -> str:
    """The kind of the destination a `post_message` record names. The record joins the
    destination and the text with commas, and a group's name may hold one: the longest
    cut that names a destination is it."""
    pieces = params.split(",")
    for cut in range(len(pieces), 0, -1):
        destination = Destinations.find(",".join(pieces[:cut]).strip().strip("\"'`)"))
        if destination:
            return destination.kind
    trace(f"Reactivation: no destination answers to the start of post_message({params[:40]}…)")
    return UNKNOWN_DESTINATION


def record(step: Step, executed_commands: list[str], now: Optional[float] = None) -> str:
    """The check that carried *step* has ended: the step is spent — sent when the check posted
    a message, declined when it did not — and reported. A step whose silence the owner ended
    meanwhile is reported and not kept: the schedule it belonged to is gone. Returns the outcome."""
    now = time.time() if now is None else now
    via = sorted(set(posted_kinds(executed_commands)))
    outcome = OUTCOME_SENT if via else OUTCOME_DECLINED
    current = _spend(step, outcome, via, now)
    info(f"Reactivation: step {step.number} ({step.kind}) {outcome}"
         + (f" to {', '.join(via)}" if via else " — the check posted nothing")
         + ("" if current else "; the owner spoke meanwhile, the schedule has started over"))
    return outcome


def _spend(step: Step, outcome: str, via: list[str], now: float) -> bool:
    """Keep *step* as spent and report it. False when the owner spoke since the step was found:
    the schedule it belonged to is gone, and the step is reported and not kept."""
    with _lock:
        state = _load()
        current = state is not None and state["last_owner_turn"] == step.since
        if current:
            state["spent"].append({"step": step.number, "at": now, "outcome": outcome})
            _save(state)
    Telemetry.get_instance().reactivation_nudge(
        step=step.number, kind=step.kind, outcome=outcome, via=via,
        owner_channels=[name for name, _ in _channels()], inactive_hours=step.quiet_seconds / 3600)
    return current
