import json
import os
import re
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

from .LongTermMemoryCurator import LongTermMemoryCurator
from ..llm import create_llm, parse_llm_config, messages_contain_images
from ..llm.AbstractLLM import LLMRefusalError
from common.logging import trace, info, note, error, fine
from assistant import Destinations, MemoryFile, Procedures, WatchList
from assistant.Telemetry import Telemetry
from common.Paths import CACHE_DIR
from ..prompt.PromptManager import PromptManager
from ..prompt.MessageHistory import MessageHistory

SCHEDULE_FILE = os.path.join(os.path.dirname(__file__), "scheduled_wakeups.json")
# When the last periodic check started, so a restart keeps the cadence instead of starting it
# over: the first check used to come a full interval after boot, which put it 68–97 minutes
# after the one before on a day of restarts — past the hour the prompt prefix stays cached,
# and every such check re-wrote the whole prefix (28K tokens at the 1h write rate on prod).
LAST_CHECK_FILE = CACHE_DIR / "last_autonomous_check.json"

# Suppress automatic activity checks for this many seconds after a user
# interaction so we don't double-reply on top of the user's own request.
USER_INTERACTION_QUIET_PERIOD_SECONDS = 60

# A wake-up the user asked for ("in 15 seconds") waits only this long after their last request
# — the full quiet period would hold a sub-minute wake-up back by a minute or more every time.
SCHEDULED_WAKEUP_QUIET_PERIOD_SECONDS = 3

# If the in-flight counter has been non-zero for longer than this without any
# fresh user activity, assume a start/end pairing was leaked and reset it so
# periodic checks can resume. A real request lasts at most a couple minutes.
USER_INTERACTION_STALE_COUNTER_SECONDS = 10 * 60

# A wake-up that fell due while the server was down fires this long after startup — time for
# the interfaces and integrations to come up before the check runs commands through them.
MISSED_WAKEUP_STARTUP_GRACE_SECONDS = 30
# Past this, a firing wake-up tells the LLM how late it is (a restart's few seconds need no note).
WAKEUP_LATE_NOTE_THRESHOLD_SECONDS = 5 * 60
# How often the due watcher reads the watch list for a due time or a short cadence that has
# arrived — the one minute a due time is given to.
DUE_POLL_SECONDS = 60

# How often the background thread performs an autonomous activity check. The LLM
# is told about this cadence (see AUTONOMOUS_ACTION_PROMPT) so it can reason about
# how long until its next opportunity to act.
PERIODIC_CHECK_INTERVAL_SECONDS = 55 * 60  # 55 minutes start to start — stays under the 1h cache TTL
# The nightly memory curation is due from this hour: the first periodic tick after it that finds
# no dream journal written since then runs it (`LongTermMemoryCurator.last_run_at`), so a
# restart in the small hours — the nightly update's, say — delays the run to the next tick
# instead of losing the night. The process-local date keeps one process from starting it twice.
CURATION_HOUR = 4
PERIODIC_CHECK_INTERVAL_MINUTES = PERIODIC_CHECK_INTERVAL_SECONDS / 60
# A watch item is polled by the check nearest its cadence: without the half interval an
# "every 1h" item is 53 minutes old at the next check, not due, and polled every 110 minutes.
WATCH_POLL_TOLERANCE = timedelta(seconds=PERIODIC_CHECK_INTERVAL_SECONDS / 2)
# A check that finds the user busy is tried again this much later, not a whole interval
# later: waiting the full interval stretched the gap between checks to 110 minutes, past
# the hour Anthropic keeps the prompt prefix cached, and the next check re-wrote it all.
POSTPONED_CHECK_RETRY_SECONDS = 3 * 60

# caffeinate(): how often the tight loop checks, and for how long when the LLM gives no duration.
CAFFEINE_INTERVAL_MINUTES = 3
CAFFEINE_DEFAULT_DURATION_MINUTES = 60

# The response-format paragraph for a model without thinking of its own. A model that handles
# thinking internally (AbstractLLM.handles_thinking_internally) must not see it: Claude refuses
# the request as reasoning extraction.
THINKING_TAG_INSTRUCTION = (
    'Start your reply with a <thinking> tag and put your entire thinking and analysis in this tag. '
    'ALL reasoning must stay inside the <thinking> tag — once you close </thinking> you may only emit '
    'commands (or "NO").\n\n')

# Appended after the autonomous instruction when the check is a scheduled wake-up. The note
# states the fact and nothing more: a wake-up once carried an instruction to the model's future
# self (2026-09-29 to 2026-10-12), and on 2026-10-11 the fired check obeyed it and reminded Erik
# of a water-meter reading he had reported done — and whose watch item was gone — five hours
# earlier. What a wake-up is for lives in the history or on the watch list, which the check reads
# anyway.
SCHEDULED_WAKEUP_NOTE = """
## SCHEDULED WAKE-UP

This check is not the periodic one: you scheduled it yourself at {scheduled_at} with schedule_wakeup."""

# Appended after the autonomous instruction when the check runs because a watch item reached
# its due time or the next poll of a short cadence (`_run_due_watcher`). The items stand in full
# among the due items of the instruction; the note says why the check runs now. A reached due
# time stays among the due items of every check until the item is looked at (the facts line
# says how long ago it was reached), so a reminder a check failed to deliver comes back.
DUE_ITEMS_NOTE = """
## DUE WATCH ITEMS

This check is not the periodic one: the moment of these watch items has come (they stand in full among the due items above):
{lines}"""

# Appended to the note when the wake-up fires well past its time: the server was down at
# the moment (the 09:00 reminder of 2026-10-11 was dropped on the restart that evening), so
# the LLM should weigh whether the action still makes sense rather than act as if on time.
SCHEDULED_WAKEUP_LATE_NOTE = """

It was due at {due_at} and fires only now, {late} late: {cause}. Judge whether the action is still worth doing — a reminder for a moment that has passed may need rewording or an apology for the delay, a timer that elapsed long ago may be moot."""

AUTONOMOUS_ACTION_PROMPT = """
# AUTONOMOUS ACTION CHECK

You are being called to evaluate whether you should proactively run commands or notify the user based on the conversation history and current system state provided above.

This check runs automatically about every {interval_minutes:.0f} minutes. Keep this cadence in mind: it is roughly how long until your next chance to act, so if something is time-sensitive and falls due before the next check, handle it now rather than waiting. If you need a tighter cadence for a specific task (e.g. a short timer or a fast-changing condition), schedule a wakeup or caffeinate instead of relying on this interval.

## Your Task

Review the conversation history and current system state to determine if any of the following conditions are met:

1. **Time-based notifications**: Did the user previously ask to be notified or have a command executed after a certain time period?
   - Examples: "Set a timer for 10 minutes", "Defrost my car in 15 minutes", "Remind me in an hour"
   - Calculate if enough time has elapsed since the request

2. **Condition-based notifications**: Did the user ask to be notified when a specific condition occurs?
   - Examples: "Let me know when it's 15 minutes until rain", "Notify me when Sam is 10 minutes away", "Tell me when the pool reaches 28 degrees"
   - Check current sensor data, weather forecasts, vehicle locations, etc.

3. **Urgent system alerts**: Are there critical issues requiring immediate attention?
   - Examples: Security alerts, system failures, dangerous conditions
   - Only for truly urgent matters
   
4. **Anything else creative you can think of**: 
   - Examples: the car is low on battery and electricity is cheap, Start defrost on the car since its cold outside and user is heading out soon
      
No need to inform the user about Telegram or iMessage messages s/he has received - those have already been pushed to the user.

## Important Rules

- **Calendar events are EXCLUDED**: The user's iPhone automatically handles calendar reminders. Do NOT notify about calendar events.
- **Check timestamps**: Pay close attention to when requests were made (look for time indicators like "10m ago", "2h ago") and the current time.
- **Verify conditions carefully**: Use actual system data (weather forecasts, vehicle status, sensor readings) to evaluate conditions.
- **OK to run commands quietly in the background**: If you need to run a command to check something for the user in the background, feel free to do that but don't interrupt the user by speaking each time.
- **Continue monitoring tasks**: If the conversation history shows the user asked you to monitor something (e.g. watch for emails, track a delivery, check a condition), you MUST continue checking on every autonomous action call until the user explicitly tells you to stop or the condition is met. Do not stop checking just because time has passed or previous checks returned no results. The caffeinate timer ending does NOT mean you should stop monitoring.
- **Watch list polls**: The items under "Watch list polls due" below are the ones whose check cadence has elapsed since they were last checked. Poll each of them now, then stamp it: `checked_watch` when nothing changed, `save_watch` with the new state when something did. Skip the stamp and the poll is lost — nothing else records that you looked. Any other item in the WATCH LIST above can be read in full with `load_watch` when you need it.

## Watch list polls due

The WATCH LIST above is only an index. These are the items whose poll is due, with their whole notes, cadences and check stamps:

{watch_list}{watch_toolkit}

## Response Format

{thinking_tag_instruction}Output only commands. Feel free to do chains of commands (send one or several commands, get reply back then send another batch of commands and so on) — just running commands does not create noise for the user.

Anything you want said goes through `post_message`: to `owner` for the user's chat(s) (formatted by the rules of the 
"Text Chat" section), or to a group by its name, or to a room by its name when someone is there and the matter is worth a spoken line (a request 
made by voice, about something in the room, etc). The destinations now: {destinations}. No other text outside commands.

Nothing to say: output "NO" — after your commands, or alone when there is nothing to do either. "NO" is a verdict, not a message; it is never sent anywhere.

Now analyze the conversation and system state above and respond accordingly.
"""


def _describe_duration(seconds: float) -> str:
    """'3 min', '2 h 15 min' or '1 d 4 h' — how late a wake-up is, in the note and the log."""
    return WatchList.describe_duration(timedelta(seconds=seconds))


class AutonomousActionManager:
    """Periodically asks the LLM whether Floxy should act or notify the user unprompted.

    Singleton with a daemon thread on PERIODIC_CHECK_INTERVAL_SECONDS; checks are suppressed
    while a user request is in flight, and scheduled wake-ups persist in SCHEDULE_FILE across
    restarts — one that fell due while the server was down fires right after startup.
    """
    _instance: Optional['AutonomousActionManager'] = None

    @classmethod
    def get_instance(cls, prompt_manager: PromptManager = None, test_mode: bool = False) -> 'AutonomousActionManager':
        if cls._instance is None:
            cls._instance = cls(prompt_manager, test_mode)
        return cls._instance

    @classmethod
    def current(cls) -> Optional['AutonomousActionManager']:
        """The running manager, or None — get_instance would construct one."""
        return cls._instance

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    def __init__(self, prompt_manager: PromptManager, test_mode: bool = False):
        info("Starting autonomous action manager...")
        llm_name, model = parse_llm_config('autonomous_action')
        self.llm = create_llm(llm_name, model)
        self.llm.initialize()
        self.llm_vision = None  # lazily created when a check actually carries images
        self.prompt_manager = prompt_manager
        self.running = True
        self.activity_check_lock = threading.Lock()
        self._user_interaction_lock = threading.Lock()
        self._user_requests_in_flight = 0
        self.last_user_interaction_at: float = 0.0
        # One caffeine loop at most: a repeated caffeinate() moves the deadline, it never
        # starts a second loop (67 of them once ran side by side, each checking every 3 min).
        self._caffeine_lock = threading.Lock()
        self._caffeine_until: float = 0.0
        self._caffeine_interval_minutes: float = CAFFEINE_INTERVAL_MINUTES
        self._caffeine_thread: Optional[threading.Thread] = None
        self.thread = threading.Thread(target=self._run_periodic_check, daemon=True)
        # The (item, moment) pairs the due watcher has fired for: once each, whatever the LLM
        # then does with the item. In memory only — a restart fires a reached moment again.
        self._fired_due: set[tuple[str, datetime]] = set()
        self._next_due_logged: Optional[datetime] = None   # the watcher logs the next moment when it changes
        self.due_thread = threading.Thread(target=self._run_due_watcher, daemon=True, name="DueWatcher")
        if not test_mode:
            self.thread.start()
            self.due_thread.start()
            self._restore_scheduled_wakeups()
        self.test_mode = test_mode

    def _get_vision_llm(self):
        if self.llm_vision is None:
            name, model = parse_llm_config('vision')
            llm = create_llm(name, model)
            llm.initialize()
            self.llm_vision = llm
            info("AutonomousActionManager: initialized vision LLM for image-bearing check")
        return self.llm_vision

    @staticmethod
    def _action_prompt(llm, prompt_manager: PromptManager | None = None) -> str:
        """The autonomous instruction for *llm*: with the <thinking>-tag paragraph unless the
        model thinks on its own, in which case asking would get the check refused. With a
        *prompt_manager* the instruction is rendered like a toolkit reference, so the watch
        toolkit it carries and any command in it speak the active command format."""
        internal = llm.handles_thinking_internally() is True
        if internal:
            trace(f"Autonomous prompt without the <thinking> tag paragraph: {llm.name()} thinks on its own")
        due = WatchList.render_due(tolerance=WATCH_POLL_TOLERANCE)
        prompt = AUTONOMOUS_ACTION_PROMPT.format(
            interval_minutes=PERIODIC_CHECK_INTERVAL_MINUTES,
            watch_list=due,
            watch_toolkit=AutonomousActionManager._watch_toolkit(due),
            destinations=Destinations.describe(),
            thinking_tag_instruction="" if internal else THINKING_TAG_INSTRUCTION)
        if prompt_manager is None:
            return prompt
        return prompt_manager.render_template(prompt)

    @staticmethod
    def _watch_toolkit(due: str) -> str:
        """The watch toolkit's reference, under the due items when there are any: the check
        must stamp or rewrite each of them, and the prompt documents only `load_watch`. It
        rides in the instruction — the check's own turn, nothing persisted, no load round —
        and a check with nothing due loads the toolkit itself if it wants an item saved."""
        if due == WatchList.NOTHING_DUE:
            return ""
        from assistant.commands import Toolkits
        toolkit = Toolkits.by_name(Toolkits.WATCH_TOOLKIT)
        trace(f"Autonomous prompt carries the {toolkit.name} toolkit: items are due (rendered with the instruction)")
        return "\n\nThe commands to stamp or rewrite them:\n\n" + Toolkits.render(toolkit)

    def _llm_for(self, messages):
        """Pick the vision tier when the outgoing messages carry image blocks
        (e.g. a command returned camera snapshots), else the autonomous tier."""
        if messages_contain_images(messages):
            trace("Autonomous check carries images — using vision tier")
            return self._get_vision_llm()
        return self.llm

    def _think(self, messages, volatile_tail: int = 0, follow_up: bool = False) -> str:
        """One blocking LLM turn. A safety refusal (Fable refuses these checks in bursts,
        deterministically) is handed to the tier's refusal_fallback() once, as
        DialogManager does for user turns; without one, or when the fallback refuses
        too, the check yields an empty result and the next interval tries again.
        `volatile_tail` is 1 for the first round: its trailing turn is the clock note and
        the instruction folded onto the history, which no later request sends. A follow-up
        round ends in the reply and the command result, which the next round repeats and no
        later check reads — `follow_up` lets the LLM cache it for its short tier."""
        llm = self._llm_for(messages)
        # The action prompt was built for the model of the first round; a follow-up round
        # that carries images goes to the vision tier, which may think on its own.
        if llm.handles_thinking_internally() is True:
            stripped = MessageHistory.remove_from_user_turns(messages, THINKING_TAG_INSTRUCTION)
            if stripped:
                info(f"{llm.name()} thinks on its own — <thinking> tag paragraph removed from the autonomous prompt")
        try:
            return llm.think_with_messages(messages, volatile_tail=volatile_tail, follow_up=follow_up)
        except LLMRefusalError:
            fallback = llm.refusal_fallback()
            if fallback is None:
                info(f"{llm.name()} refused the autonomous check (safety filter) and has no fallback — skipping.")
                return ""
            info(f"{llm.name()} refused the autonomous check (safety filter) — falling back to {fallback.name()}.")
            fallback.initialize()
            try:
                return fallback.think_with_messages(messages, volatile_tail=volatile_tail, follow_up=follow_up)
            except LLMRefusalError:
                info(f"{fallback.name()} refused the autonomous check too — skipping.")
                return ""

    def notify_user_interaction_started(self):
        with self._user_interaction_lock:
            self._user_requests_in_flight += 1
            self.last_user_interaction_at = time.time()

    def notify_user_interaction_ended(self):
        with self._user_interaction_lock:
            if self._user_requests_in_flight > 0:
                self._user_requests_in_flight -= 1
            self.last_user_interaction_at = time.time()

    def is_user_interaction_active(self, quiet_seconds: float = USER_INTERACTION_QUIET_PERIOD_SECONDS) -> bool:
        with self._user_interaction_lock:
            if self._user_requests_in_flight > 0:
                # Self-heal a leaked counter: if no interaction has refreshed
                # the timestamp for a long time, the start/end pairing was lost
                # (e.g. client disconnect before the response generator closed).
                age = time.time() - self.last_user_interaction_at
                if age > USER_INTERACTION_STALE_COUNTER_SECONDS:
                    error(
                        f"Resetting stale user-interaction counter "
                        f"({self._user_requests_in_flight} in flight, "
                        f"last activity {age:.0f}s ago) — assuming leaked pairing."
                    )
                    self._user_requests_in_flight = 0
                    return False
                return True
            return (time.time() - self.last_user_interaction_at) < quiet_seconds

    def _run_periodic_check(self):
        last_memory_curation_date = None

        next_check_at = self._first_check_at()
        while self.running:
            time.sleep(max(0.0, next_check_at - time.time()))

            now = datetime.now()
            if now.hour >= CURATION_HOUR and last_memory_curation_date != now.date():
                last_memory_curation_date = now.date()
                if self._nightly_curation_due(now):
                    # On a thread of its own: the batch it waits for can be over an hour away,
                    # and this loop has other things to check meanwhile.
                    threading.Thread(target=self._run_memory_curation, kwargs={'batch': True},
                                     daemon=True, name="MemoryCuration").start()

            if self.prompt_manager.is_night_mode_active():
                next_check_at = time.time() + PERIODIC_CHECK_INTERVAL_SECONDS
                continue

            if self.is_user_interaction_active():
                next_check_at = time.time() + POSTPONED_CHECK_RETRY_SECONDS
                info(f"Postponing periodic activity check {POSTPONED_CHECK_RETRY_SECONDS // 60} min "
                     f"— user interaction active or recent.")
                continue

            # Timed from the start of the check, so its own duration does not stretch the
            # cadence past the hour the prompt prefix stays cached.
            next_check_at = time.time() + PERIODIC_CHECK_INTERVAL_SECONDS
            self._write_last_check(time.time())
            try:
                self.activity_check()
            except Exception as e:
                # One failed check (the API out of credits, say) must not end the periodic
                # loop: on 2026-10-04 it did, and no check ran again until the next restart.
                error(f"Periodic activity check failed: {type(e).__name__}: {e}")
            trace(f"Next periodic activity check at {time.strftime('%H:%M', time.localtime(next_check_at))}")

    @staticmethod
    def _first_check_at(now: float | None = None) -> float:
        """When the first periodic check of this process runs: an interval after the last one
        the previous process started, so a restart neither resets the cadence nor stretches it
        past the cache hour. Without a record (first boot, a wiped cache) the interval counts
        from now; a record older than an interval fires after the startup grace, the moment the
        interfaces are up — the cache is cold by then anyway, but the LLM was promised a check
        every interval."""
        now = time.time() if now is None else now
        last = AutonomousActionManager._read_last_check()
        if last is None:
            trace("No record of the last periodic check — the first one is an interval from now")
            return now + PERIODIC_CHECK_INTERVAL_SECONDS
        at = max(last + PERIODIC_CHECK_INTERVAL_SECONDS, now + MISSED_WAKEUP_STARTUP_GRACE_SECONDS)
        info(f"Last periodic check started {_describe_duration(now - last)} ago — "
             f"the next one at {time.strftime('%H:%M', time.localtime(at))}")
        return at

    @staticmethod
    def _read_last_check() -> float | None:
        try:
            with open(LAST_CHECK_FILE) as f:
                return float(json.load(f)["started_at"])
        except (FileNotFoundError, ValueError, TypeError, KeyError):
            return None
        except OSError as e:
            error(f"Failed to read the last periodic check record: {e}")
            return None

    @staticmethod
    def _write_last_check(started_at: float):
        try:
            with open(LAST_CHECK_FILE, "w") as f:
                json.dump({"started_at": started_at}, f)
        except OSError as e:
            error(f"Failed to record the periodic check start: {e}")

    def activity_check(self, prompt: str | None = None, wakeup: dict | None = None,
                       due: list[tuple[WatchList.Item, datetime]] | None = None) -> str:
        """One autonomous check. *wakeup* is the schedule-file entry when the check is a
        scheduled wake-up, *due* the watch items (with the moment each reached) when the due
        watcher runs it: the instruction says so (SCHEDULED_WAKEUP_NOTE, DUE_ITEMS_NOTE)."""
        if not self.activity_check_lock.acquire(timeout=120):
            error("Could not acquire activity_check_lock — another check is stuck")
            return ""
        try:
            return self._activity_check_inner(prompt, wakeup, due)
        finally:
            self.activity_check_lock.release()

    @staticmethod
    def _due_note(due: list[tuple[WatchList.Item, datetime]], now: datetime = None) -> str:
        now = now or datetime.now()
        lines = []
        for item, moment in due:
            if WatchList.due_moment(item) == moment:
                lines.append(f"- `{item.name}` — due {item.due}, reached {WatchList.describe_duration(now - moment)} ago")
            else:
                lines.append(f"- `{item.name}` — due for its check (every {item.every})")
        return DUE_ITEMS_NOTE.format(lines="\n".join(lines))

    @staticmethod
    def _wakeup_note(wakeup: dict) -> str:
        scheduled_at = time.strftime("%H:%M", time.localtime(wakeup["scheduled_at"]))
        note = SCHEDULED_WAKEUP_NOTE.format(scheduled_at=scheduled_at)
        late_seconds = time.time() - wakeup["at"]
        if late_seconds > WAKEUP_LATE_NOTE_THRESHOLD_SECONDS:
            due_at = time.strftime("%Y-%m-%d %H:%M", time.localtime(wakeup["at"]))
            cause = ("the server was down at the time" if wakeup.get("missed")
                     else "it was held back while you were busy with the user")
            note += SCHEDULED_WAKEUP_LATE_NOTE.format(due_at=due_at, late=_describe_duration(late_seconds), cause=cause)
        return note

    def _activity_check_inner(self, prompt: str | None, wakeup: dict | None = None,
                              due: list[tuple[WatchList.Item, datetime]] | None = None) -> str:
            if wakeup:
                info(f"Check for autonomous action (wake-up scheduled at "
                     f"{time.strftime('%H:%M', time.localtime(wakeup['scheduled_at']))})...")
            elif due:
                info("Check for autonomous action (watch items reached their moment: "
                     + ", ".join(item.name for item, _ in due) + ")...")
            else:
                info("Check for autonomous action...")
            is_real_run = prompt is None
            history_messages = []
            if is_real_run:
                prompt = self.prompt_manager.prompt
                prompt = prompt.replace("[WHISPER]", "").replace("[THINKING]", "")
                history_messages = self.prompt_manager.message_history.get_history_messages()

            # Build the full context once: system prompt + live conversation history
            # + the autonomous-action instruction. History used to be baked into the
            # prompt text; it is now appended as message turns, so the autonomous
            # check must include it explicitly — otherwise it evaluates "the
            # conversation history above" with no history at all, and the saved
            # prompt file would not reflect what was sent.
            messages = [{"role": "system", "content": prompt}]
            messages.extend(history_messages)
            # Inject the current time the same way get_messages() does. The clock used
            # to live in the system prompt but moved to the trailing user turn to keep
            # the prompt bytes cache-stable — so this check, which builds its own
            # message list, must add it back. Without it the LLM has no clock, yet
            # AUTONOMOUS_ACTION_PROMPT asks it to compare elapsed time against the
            # current time (e.g. "remind me in 10m"). Real runs only — provided-prompt
            # test runs must stay free of wall-clock state.
            if is_real_run:
                MessageHistory.append_user_turn(messages, self.prompt_manager.current_time_note())
            MessageHistory.append_user_turn(messages, self._action_prompt(self._llm_for(messages), self.prompt_manager))
            if wakeup:
                MessageHistory.append_user_turn(messages, self._wakeup_note(wakeup))
            if due:
                MessageHistory.append_user_turn(messages, self._due_note(due))

            if is_real_run:
                self.prompt_manager.save_prompt_to_file(messages)

            start_time = time.time()
            result = self._think(messages, volatile_tail=1)
            fine(f"Raw result:\n {result}")
            trace(f"Thought for {time.time() - start_time:.2f}s")

            from assistant.commands.CommandProcessor import CommandProcessor
            command_processor = CommandProcessor(self.test_mode)
            stray = self._run_commands(result, command_processor)

            # Wait for queued commands to be picked up and processed
            # (prevents race where we check flags before the worker thread sets them)
            self._join_queue_with_timeout(command_processor.command_queue, timeout=30)

            if command_processor.command_being_processed:
                trace("Waiting for command to complete...")
                deadline = time.time() + 60
                while command_processor.command_being_processed and time.time() < deadline:
                    time.sleep(0.01)
                if command_processor.command_being_processed:
                    error("Timed out waiting for command to complete")
                else:
                    trace("Command completed.")

            # Handle background commands - loop back to LLM with results.
            # `messages` already holds system + history + the autonomous instruction;
            # append the model's reply so follow-up calls keep the full context.
            messages.append({"role": "assistant", "content": result})

            ran_commands = False
            follow_up_saved = False
            while not self.test_mode and (command_processor.command_running_in_background or
                                          command_processor.command_reply is not None):
                ran_commands = True
                info("Background command detected, circling back to LLM...")

                if command_processor.command_running_in_background:
                    trace("Waiting until background command is done...")
                    deadline = time.time() + 120
                    while command_processor.command_running_in_background and time.time() < deadline:
                        time.sleep(0.01)
                    if command_processor.command_running_in_background:
                        error("Timed out waiting for background command")
                        break
                    trace("Background command done.")

                if (command_processor.command_reply is None and
                        command_processor.command_queue.empty()):
                    info("Background command completed silently, skipping LLM follow-up.")
                    break

                if command_processor.command_reply is None:
                    trace("Waiting for command reply...")
                    deadline = time.time() + 30
                    while command_processor.command_reply is None and time.time() < deadline:
                        time.sleep(0.01)
                    if command_processor.command_reply is None:
                        error("Timed out waiting for command reply")
                        break
                    trace("Command reply received.")

                reply = command_processor.command_reply
                fine("Command reply: " + (f"[list with {len(reply)} blocks]" if isinstance(reply, list) else str(reply)))

                # Add the command result to messages and call LLM again
                if isinstance(reply, list):
                    messages.append({"role": "user", "content": reply})
                else:
                    messages.append({"role": "user", "content": f"result:\n{reply}"})
                command_processor.clear_command_reply()  # Reset for next iteration
                # Save the *initial* (pre-loop) result on the first iteration only.
                # On subsequent iterations the previous iteration already wrote
                # the current `result` via _save_autonomous_result_to_history.
                if not follow_up_saved:
                    self.clean_result_and_potentially_add_to_history(result)

                info("Checking with AI again...")
                start_time = time.time()
                result = self._think(messages, follow_up=True)
                trace(f"Follow-up thought for {time.time() - start_time:.2f}s")
                fine(f"Follow-up raw result:\n {result}")
                self._save_autonomous_result_to_history(result)
                follow_up_saved = True

                # Add assistant response to messages for potential next iteration
                messages.append({"role": "assistant", "content": result})

                # The same processor reads every round of the check: the command cap is per reply.
                command_processor.start_new_reply()
                stray = self._run_commands(result, command_processor)

                # Wait for queued follow-up commands to be picked up and processed
                self._join_queue_with_timeout(command_processor.command_queue, timeout=30)

                if command_processor.command_being_processed:
                    trace("Waiting for follow-up command to complete...")
                    deadline = time.time() + 60
                    while command_processor.command_being_processed and time.time() < deadline:
                        time.sleep(0.01)
                    if command_processor.command_being_processed:
                        error("Timed out waiting for follow-up command to complete")
                    else:
                        trace("Follow-up command completed.")

            if ran_commands and not follow_up_saved:
                self._save_autonomous_result_to_history(result)
            elif not ran_commands:
                self.clean_result_and_potentially_add_to_history(result)
            # The check emits commands only; what it wants said goes through post_message. "NO"
            # is the explicit verdict that there was nothing to do; any other text left over is
            # delivered nowhere — the trace is there to see it in the log if the model ever
            # strays from the format.
            self._drop_stray_text(stray)
            return result

    @staticmethod
    def _run_commands(result: str, command_processor) -> str:
        """Feed the LLM reply (thinking stripped) through the CommandProcessor as one
        whitespace-preserving chunk, as DialogManager's stream feed does, so the commands in it
        run. Returns the text the processor did not consume — only for `_drop_stray_text`."""
        result_no_thinking = re.sub(r'<thinking>.*?</thinking>', '', result, flags=re.DOTALL)
        stray = command_processor.process(result_no_thinking.replace("`", ""), None) or ""
        remaining = command_processor.flush_buffer(None)
        if remaining:
            stray += remaining
        return stray

    # The sentinel the check answers with when there is nothing to say and nothing to do. A
    # model is bad at outputting nothing at all, so the instruction asks for this one word
    # instead; it is recognised with any punctuation or case a flaky model wraps it in.
    NO_SENTINEL = "NO"

    @classmethod
    def _is_no(cls, text: str) -> bool:
        return re.sub(r'[^A-Za-z]', '', text or "").upper() == cls.NO_SENTINEL

    @classmethod
    def _drop_stray_text(cls, stray: str) -> None:
        text = re.sub(r'<thinking>.*?</thinking>', '', stray or "", flags=re.DOTALL)
        text = re.sub(r'\[[^\]]*\]', '', text).strip()
        if cls._is_no(text):
            trace("Autonomous check answered NO: nothing to say, nothing to do.")
        elif text:
            trace(f"Autonomous check left {len(text)} chars of text after its commands — dropped, post_message is "
                  f"how to say something: {text[:120]!r}")

    @staticmethod
    def _join_queue_with_timeout(q, timeout: float):
        with q.all_tasks_done:
            while q.unfinished_tasks:
                if not q.all_tasks_done.wait(timeout=timeout):
                    error("Timed out waiting for command queue to drain")
                    return

    def clean_result_and_potentially_add_to_history(self, result: str) -> str:
        # Filter out XML tags (including content) and square brackets
        result_cleaned = re.sub(r'<thinking>.*?</thinking>', '', result, flags=re.DOTALL)
        result_cleaned = re.sub(r'<[^>]*>', '', result_cleaned)
        result_cleaned = re.sub(r'\[[^\]]*\]', '', result_cleaned).strip()
        if result_cleaned and not self._is_no(result_cleaned):
            self.prompt_manager.update_history(None, None, result_cleaned)
        return result_cleaned

    def _save_autonomous_result_to_history(self, result: str):
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', result, flags=re.DOTALL)
        thinking_text = thinking_match.group(1).strip() if thinking_match else ""

        visible = re.sub(r'<thinking>.*?</thinking>', '', result, flags=re.DOTALL)
        visible = re.sub(r'<[^>]*>', '', visible)
        visible = re.sub(r'\[[^\]]*\]', '', visible).strip()

        parts = []
        if thinking_text:
            info(f"Thinking: {thinking_text}")
            # Store with the same <thinking> tag the model uses in its own
            # output. Earlier versions wrote "[Thinking] ..." here, which the
            # model then mimicked in fresh responses — an unclosed prefix the
            # streaming parser couldn't strip.
            parts.append(f"<thinking>{thinking_text}</thinking>")
        if visible and not self._is_no(visible):
            parts.append(visible)

        if parts:
            self.prompt_manager.update_history(None, None, "\n".join(parts))

    # --- the due watcher ------------------------------------------------------------------

    def _run_due_watcher(self):
        """Reads the watch list every DUE_POLL_SECONDS and runs a check when an item's due time
        or the next poll of a cadence shorter than the periodic interval has arrived. The first
        read waits the startup grace, so a moment reached while the server was down fires once
        the interfaces are up — the item file is the schedule, nothing else persists."""
        time.sleep(MISSED_WAKEUP_STARTUP_GRACE_SECONDS)
        while self.running:
            try:
                self._poll_due()
            except Exception as e:
                error(f"Due watcher failed: {type(e).__name__}: {e}")
            time.sleep(DUE_POLL_SECONDS)

    def _poll_due(self, now: datetime = None) -> bool:
        """One read of the watch list: fires a check for the reached moments not fired yet and
        says so; False when it fired nothing. A moment is fired once (`_fired_due`) — an item
        the LLM leaves unlooked-at after that is the periodic checks' business, which carry it
        as a due item until a look. Night mode leaves the moment unfired, so it fires in the
        morning; a user in flight leaves it for the next read a minute on."""
        now = now or datetime.now()
        pending = WatchList.pending(now, within=timedelta(seconds=PERIODIC_CHECK_INTERVAL_SECONDS))
        # An item earns several moments at once (its due time and its cadence's next poll), each
        # fired once; a moment no longer pending was looked at and cannot come back.
        self._fired_due &= {(item.name, moment) for item, moment in pending}
        reached = [(item, moment) for item, moment in pending
                   if moment <= now and (item.name, moment) not in self._fired_due]
        if not reached:
            ahead = [moment for _, moment in pending if moment > now]
            next_due = min(ahead) if ahead else None
            if next_due != self._next_due_logged:
                self._next_due_logged = next_due
                trace("Due watcher: " + (f"next moment at {next_due:%Y-%m-%d %H:%M}" if next_due else "no moment ahead"))
            return False
        names = ", ".join(item.name for item, _ in reached)
        if self.prompt_manager.is_night_mode_active():
            info(f"Due watcher: {names} reached — left for the morning, Night mode is active")
            return False
        if self.is_user_interaction_active():
            info(f"Due watcher: {names} reached — user interaction active or recent, trying again in a minute")
            return False
        self._fired_due.update((item.name, moment) for item, moment in reached)
        info(f"Due watcher: firing a check for {names}")
        self.activity_check(due=reached)
        return True

    def schedule(self, minutes: float) -> None:
        """Wake in *minutes*. What the wake is for stays in the history or on the watch list —
        a wake-up carries no instruction of its own (see SCHEDULED_WAKEUP_NOTE)."""
        wake_at = time.time() + minutes * 60
        info(f"Scheduling Autonomous Action Manager to wake in {minutes} minutes...")
        wakeup = {"at": wake_at, "scheduled_at": time.time()}
        self._save_wakeup(wakeup)
        self._start_wakeup_thread(wakeup)

    def _start_wakeup_thread(self, wakeup: dict):
        wake_at = wakeup["at"]

        def _delayed_check():
            remaining = wake_at - time.time()
            if remaining > 0:
                time.sleep(remaining)
            self._remove_wakeup(wake_at)
            if self.prompt_manager.is_night_mode_active():
                info("Scheduled wakeup skipped — Night mode is active")
                return
            # Hold the wake-up only while a request is in flight or has just ended, so it
            # does not pile a second reply on top of the user's own — but no longer: the
            # user asked for this moment.
            while self.running and self.is_user_interaction_active(SCHEDULED_WAKEUP_QUIET_PERIOD_SECONDS):
                info("Scheduled wakeup deferred — user interaction active or recent. Retrying shortly.")
                time.sleep(SCHEDULED_WAKEUP_QUIET_PERIOD_SECONDS)
            self.activity_check(wakeup=wakeup)

        thread = threading.Thread(target=_delayed_check, daemon=True)
        thread.start()

    def _save_wakeup(self, wakeup: dict):
        self._write_wakeups(self._load_wakeups() + [wakeup])

    def _remove_wakeup(self, wake_at: float):
        self._write_wakeups([w for w in self._load_wakeups() if w["at"] != wake_at])

    @staticmethod
    def _write_wakeups(wakeups: list[dict]):
        try:
            with open(SCHEDULE_FILE, "w") as f:
                json.dump(wakeups, f)
        except Exception as e:
            error(f"Failed to save scheduled wakeups: {e}")

    @staticmethod
    def _load_wakeups() -> list[dict]:
        """The pending wake-ups on disk. A bare timestamp (the file's oldest shape) is read as
        a wake-up scheduled at an unknown time; the `reason` an older version wrote is dropped."""
        try:
            with open(SCHEDULE_FILE, "r") as f:
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        wakeups = []
        for entry in raw:
            if isinstance(entry, dict):
                wakeups.append({"at": entry["at"], "scheduled_at": entry.get("scheduled_at", entry["at"])})
            else:
                wakeups.append({"at": float(entry), "scheduled_at": float(entry)})
        return wakeups

    def _restore_scheduled_wakeups(self):
        """The wake-ups on disk pick up where they were: a pending one waits for its time, one
        that fell due while the server was down fires after the startup grace, oldest first.
        A missed one stays in the file until it has fired, so a second crash still keeps it."""
        wakeups = self._load_wakeups()
        now = time.time()
        active = [w for w in wakeups if w["at"] > now]
        missed = sorted((w for w in wakeups if w["at"] <= now), key=lambda w: w["at"])
        if not wakeups:
            trace("No scheduled wake-ups to restore")
        for wakeup in active:
            remaining = (wakeup["at"] - now) / 60
            info(f"Restoring scheduled wakeup in {remaining:.1f} minutes")
            self._start_wakeup_thread(wakeup)
        if missed:
            oldest = time.strftime("%Y-%m-%d %H:%M", time.localtime(missed[0]["at"]))
            info(f"{len(missed)} scheduled wakeup(s) fell due while the server was down (oldest {oldest}) — "
                 f"firing them in {MISSED_WAKEUP_STARTUP_GRACE_SECONDS} s")
            threading.Thread(target=self._fire_missed_wakeups, args=(missed,), daemon=True,
                             name="MissedWakeups").start()

    def _fire_missed_wakeups(self, missed: list[dict]):
        """Run the checks of *missed* one after the other, oldest first, after the startup grace
        — the same gating as a wake-up that fires on time, and each is removed from the file as
        its turn comes."""
        time.sleep(MISSED_WAKEUP_STARTUP_GRACE_SECONDS)
        for wakeup in missed:
            if not self.running:
                return
            self._remove_wakeup(wakeup["at"])
            late = _describe_duration(time.time() - wakeup["at"])
            if self.prompt_manager.is_night_mode_active():
                info(f"Missed wakeup ({late} late) skipped — Night mode is active")
                continue
            while self.running and self.is_user_interaction_active(SCHEDULED_WAKEUP_QUIET_PERIOD_SECONDS):
                info("Missed wakeup deferred — user interaction active or recent. Retrying shortly.")
                time.sleep(SCHEDULED_WAKEUP_QUIET_PERIOD_SECONDS)
            info(f"Firing missed wakeup {late} late")
            try:
                self.activity_check(wakeup={**wakeup, "missed": True})
            except Exception as e:
                error(f"Missed wakeup check failed: {type(e).__name__}: {e}")

    def caffeinate(self, duration_minutes: float = CAFFEINE_DEFAULT_DURATION_MINUTES,
                   interval_minutes: float = CAFFEINE_INTERVAL_MINUTES):
        """Check every *interval_minutes* until *duration_minutes* from now. Idempotent: a call
        while caffeinated only moves the deadline of the one running loop."""
        with self._caffeine_lock:
            self._caffeine_until = time.time() + duration_minutes * 60
            self._caffeine_interval_minutes = interval_minutes
            until = time.strftime("%H:%M", time.localtime(self._caffeine_until))
            if self._caffeine_thread is not None and self._caffeine_thread.is_alive():
                info(f"Already caffeinated — deadline moved to {until} (every {interval_minutes:g} min).")
                return
            info(f"Caffeinating: waking every {interval_minutes:g} minutes until {until}...")
            self._caffeine_thread = threading.Thread(target=self._caffeine_loop, daemon=True)
            self._caffeine_thread.start()

    def _caffeine_loop(self):
        wakeups = 0
        while self.running:
            time.sleep(self._caffeine_interval_minutes * 60)
            if not self.running:
                break
            remaining = self._caffeine_until - time.time()
            if remaining <= 0:
                break
            wakeups += 1
            info(f"Caffeine wakeup #{wakeups}, {remaining / 60:.0f} min left")
            if self.is_user_interaction_active():
                info("Skipping caffeine wakeup — user interaction active or recent.")
                continue
            self.activity_check()
        info(f"Caffeine wore off after {wakeups} wakeup(s).")

    @staticmethod
    def _nightly_curation_due(now: datetime) -> bool:
        """Whether tonight's curation still has to run: not when a dream journal was written
        since today's CURATION_HOUR (a run before a restart), nor while the first-run pipeline
        has yet to build the first bank (the nightly run would curate the stub)."""
        due_since = now.replace(hour=CURATION_HOUR, minute=0, second=0, microsecond=0)
        last_run = LongTermMemoryCurator.last_run_at()
        if last_run is not None and last_run >= due_since:
            info(f"Rem: the bank was curated at {last_run:%H:%M} today — no nightly run")
            return False
        from assistant import FirstRun
        if FirstRun.curation_pending():
            info("Rem: the first-run pipeline has not built the first memory bank yet — skipping the nightly run")
            return False
        if now.hour > CURATION_HOUR:
            info(f"Rem: no curation since {due_since:%H:%M} today — running it now, {now:%H:%M}")
        return True

    def _run_memory_curation(self, period_start=None, period_end=None, batch: bool = False):
        """Curate the memory bank — nightly over the default lookback, or over an explicit period.
        The completion lands in the dialog history as a system row. `batch` lets the run wait
        for the half-price batch API (the nightly run; nobody is waiting for it)."""
        note("Rem: falling asleep — starting memory curation...")
        try:
            ran = LongTermMemoryCurator.run(period_start, period_end, history=self.prompt_manager.message_history,
                                            batch=batch)
        except Exception as e:
            import traceback
            error(f"Memory curation failed: {e}")
            trace(traceback.format_exc())
            return
        if ran is None:
            trace("Rem: the curation was skipped, no memory size for the telemetry")
            return
        self._report_memory_size()

    @staticmethod
    def _report_memory_size():
        """The beta telemetry's memory event: the size of the bank as the curation left it (as
        it stood, when the period was empty) and the number of watch items and procedures."""
        try:
            bank = MemoryFile.read()
            words, watch_items, procedures = (LongTermMemoryCurator.word_count(bank), len(WatchList.items()),
                                              len(Procedures.names()))
            info(f"Rem: memory is {words} words ({len(bank)} chars), {watch_items} watch item(s), "
                 f"{procedures} procedure(s)")
            Telemetry.get_instance().memory_size(words, len(bank), watch_items, procedures)
        except Exception as e:
            error(f"Memory size telemetry failed: {type(e).__name__}: {e}")

    def stop(self):
        self.running = False
        for thread in (self.thread, self.due_thread):
            if thread.is_alive():
                thread.join(timeout=1)
