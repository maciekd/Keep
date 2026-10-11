import json
import os
import re
import threading
import time
from datetime import timedelta
from typing import Optional

from .LongTermMemoryCurator import LongTermMemoryCurator
from ..llm import create_llm, parse_llm_config, messages_contain_images
from ..llm.AbstractLLM import LLMRefusalError
from common.logging import trace, info, note, error, fine
from assistant import MemoryFile, Procedures, WatchList
from assistant.Telemetry import Telemetry
from assistant.Notifier import notifier
from ..prompt.PromptManager import PromptManager
from ..prompt.MessageHistory import MessageHistory

SCHEDULE_FILE = os.path.join(os.path.dirname(__file__), "scheduled_wakeups.json")

# Suppress automatic activity checks for this many seconds after a user
# interaction so we don't double-reply on top of the user's own request.
USER_INTERACTION_QUIET_PERIOD_SECONDS = 60

# If the in-flight counter has been non-zero for longer than this without any
# fresh user activity, assume a start/end pairing was leaked and reset it so
# periodic checks can resume. A real request lasts at most a couple minutes.
USER_INTERACTION_STALE_COUNTER_SECONDS = 10 * 60

# A wake-up that fell due while the server was down fires this long after startup — time for
# the interfaces and integrations to come up before the check runs commands through them.
MISSED_WAKEUP_STARTUP_GRACE_SECONDS = 30
# Past this, a firing wake-up tells the LLM how late it is (a restart's few seconds need no note).
WAKEUP_LATE_NOTE_THRESHOLD_SECONDS = 5 * 60

# How often the background thread performs an autonomous activity check. The LLM
# is told about this cadence (see AUTONOMOUS_ACTION_PROMPT) so it can reason about
# how long until its next opportunity to act.
PERIODIC_CHECK_INTERVAL_SECONDS = 55 * 60  # 55 minutes start to start — stays under the 1h cache TTL
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
    'commands or the final Telegram message (or "NO").\n\n')

# Appended after the autonomous instruction when the check is a scheduled wake-up: the
# exchange that scheduled it may have been trimmed from the history window by then, so the
# wake carries its own reason (the 30-minutes-before-launch reminder of 2026-09-28 fired twice
# into a window that no longer mentioned the launch, and the model found nothing to do).
SCHEDULED_WAKEUP_NOTE = """
## SCHEDULED WAKE-UP

This check is not the periodic one: you scheduled it yourself at {scheduled_at} with schedule_wakeup, for:
> {reason}

The conversation that led to it may no longer be in the history above. Act on the reason now — it is why you are awake."""

SCHEDULED_WAKEUP_NOTE_NO_REASON = """
## SCHEDULED WAKE-UP

This check is not the periodic one: you scheduled it yourself at {scheduled_at} with schedule_wakeup, without recording why. Look for the pending timer, reminder or follow-up in the history above and act on it now."""

# Appended to either note when the wake-up fires well past its time: the server was down at
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
- **Watch list polls**: The items under "Watch list polls due" below are the ones whose check cadence has elapsed since they were last checked. Poll each of them now, then stamp it: `checked_watch(name)` when nothing changed, `save_watch(...)` with the new state when something did. Skip the stamp and the poll is lost — nothing else records that you looked. Any other item in the WATCH LIST above can be read in full with `load_watch(name)` when you need it.

## Watch list polls due

The WATCH LIST above is only an index. These are the items whose poll is due, with their whole notes, cadences and check stamps:

{watch_list}

## Response Format

{thinking_tag_instruction}If you should execute commands: output any commands you need to execute (using proper command syntax). Feel free to do chains of commands (send one command, get reply back then send another command and so on). Feel free to use commands if needed - just running commands does not create noise for the user.

Lastly, if you want a Telegram message to be sent to user: output the message (after any commands). Make sure to use the formatting rules from the "Text Chat" section. If no message should be sent then output "NO" (after any commands).

Now analyze the conversation and system state above and respond accordingly.
"""


def _describe_duration(seconds: float) -> str:
    """'3 min', '2 h 15 min' or '1 d 4 h' — how late a wake-up is, in the note and the log."""
    minutes = int(seconds // 60)
    days, minutes = divmod(minutes, 24 * 60)
    hours, minutes = divmod(minutes, 60)
    if days:
        return f"{days} d {hours} h"
    if hours:
        return f"{hours} h {minutes} min"
    return f"{minutes} min"


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
        if not test_mode:
            self.thread.start()
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
    def _action_prompt(llm) -> str:
        """The autonomous instruction for *llm*: with the <thinking>-tag paragraph unless the
        model thinks on its own, in which case asking would get the check refused."""
        internal = llm.handles_thinking_internally() is True
        if internal:
            trace(f"Autonomous prompt without the <thinking> tag paragraph: {llm.name()} thinks on its own")
        return AUTONOMOUS_ACTION_PROMPT.format(
            interval_minutes=PERIODIC_CHECK_INTERVAL_MINUTES,
            watch_list=WatchList.render_due(tolerance=WATCH_POLL_TOLERANCE),
            thinking_tag_instruction="" if internal else THINKING_TAG_INSTRUCTION)

    def _llm_for(self, messages):
        """Pick the vision tier when the outgoing messages carry image blocks
        (e.g. a command returned camera snapshots), else the autonomous tier."""
        if messages_contain_images(messages):
            trace("Autonomous check carries images — using vision tier")
            return self._get_vision_llm()
        return self.llm

    def _think(self, messages) -> str:
        """One blocking LLM turn. A safety refusal (Fable refuses these checks in bursts,
        deterministically) is handed to the tier's refusal_fallback() once, as
        DialogManager does for user turns; without one, or when the fallback refuses
        too, the check yields an empty result and the next interval tries again."""
        llm = self._llm_for(messages)
        # The action prompt was built for the model of the first round; a follow-up round
        # that carries images goes to the vision tier, which may think on its own.
        if llm.handles_thinking_internally() is True:
            stripped = MessageHistory.remove_from_user_turns(messages, THINKING_TAG_INSTRUCTION)
            if stripped:
                info(f"{llm.name()} thinks on its own — <thinking> tag paragraph removed from the autonomous prompt")
        try:
            return llm.think_with_messages(messages)
        except LLMRefusalError:
            fallback = llm.refusal_fallback()
            if fallback is None:
                info(f"{llm.name()} refused the autonomous check (safety filter) and has no fallback — skipping.")
                return ""
            info(f"{llm.name()} refused the autonomous check (safety filter) — falling back to {fallback.name()}.")
            fallback.initialize()
            try:
                return fallback.think_with_messages(messages)
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
        from datetime import datetime
        last_memory_curation_date = None

        next_check_at = time.time() + PERIODIC_CHECK_INTERVAL_SECONDS
        while self.running:
            time.sleep(max(0.0, next_check_at - time.time()))

            # Check if it's 04:00 and we haven't run curation today
            now = datetime.now()
            if now.hour == 4 and last_memory_curation_date != now.date():
                from assistant import FirstRun
                if FirstRun.curation_pending():
                    info("Rem: the first-run pipeline has not built the first memory bank yet — skipping the nightly run")
                else:
                    # On a thread of its own: the batch it waits for can be over an hour away,
                    # and this loop has other things to check meanwhile.
                    threading.Thread(target=self._run_memory_curation, kwargs={'batch': True},
                                     daemon=True, name="MemoryCuration").start()
                last_memory_curation_date = now.date()

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
            try:
                self.activity_check()
            except Exception as e:
                # One failed check (the API out of credits, say) must not end the periodic
                # loop: on 2026-10-04 it did, and no check ran again until the next restart.
                error(f"Periodic activity check failed: {type(e).__name__}: {e}")
            trace(f"Next periodic activity check at {time.strftime('%H:%M', time.localtime(next_check_at))}")

    def activity_check(self, prompt: str | None = None, wakeup: dict | None = None) -> str:
        """One autonomous check. *wakeup* is the schedule-file entry when the check is a
        scheduled wake-up: its reason is appended to the instruction (see SCHEDULED_WAKEUP_NOTE)."""
        if not self.activity_check_lock.acquire(timeout=120):
            error("Could not acquire activity_check_lock — another check is stuck")
            return "NO"
        try:
            return self._activity_check_inner(prompt, wakeup)
        finally:
            self.activity_check_lock.release()

    @staticmethod
    def _wakeup_note(wakeup: dict) -> str:
        scheduled_at = time.strftime("%H:%M", time.localtime(wakeup["scheduled_at"]))
        reason = wakeup.get("reason", "")
        if reason:
            note = SCHEDULED_WAKEUP_NOTE.format(scheduled_at=scheduled_at, reason=reason)
        else:
            note = SCHEDULED_WAKEUP_NOTE_NO_REASON.format(scheduled_at=scheduled_at)
        late_seconds = time.time() - wakeup["at"]
        if late_seconds > WAKEUP_LATE_NOTE_THRESHOLD_SECONDS:
            due_at = time.strftime("%Y-%m-%d %H:%M", time.localtime(wakeup["at"]))
            cause = ("the server was down at the time" if wakeup.get("missed")
                     else "it was held back while you were busy with the user")
            note += SCHEDULED_WAKEUP_LATE_NOTE.format(due_at=due_at, late=_describe_duration(late_seconds), cause=cause)
        return note

    def _activity_check_inner(self, prompt: str | None, wakeup: dict | None = None) -> str:
            if wakeup:
                info(f"Check for autonomous action (scheduled wake-up for: {wakeup.get('reason') or 'no reason recorded'})...")
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
            MessageHistory.append_user_turn(messages, self._action_prompt(self._llm_for(messages)))
            if wakeup:
                MessageHistory.append_user_turn(messages, self._wakeup_note(wakeup))

            if is_real_run:
                self.prompt_manager.save_prompt_to_file(messages)

            start_time = time.time()
            result = self._think(messages)
            fine(f"Raw result:\n {result}")
            trace(f"Thought for {time.time() - start_time:.2f}s")

            from assistant.commands.CommandProcessor import CommandProcessor
            command_processor = CommandProcessor(self.test_mode)
            filtered = self._run_commands_and_extract_message(result, command_processor)

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
                result = self._think(messages)
                trace(f"Follow-up thought for {time.time() - start_time:.2f}s")
                fine(f"Follow-up raw result:\n {result}")
                self._save_autonomous_result_to_history(result)
                follow_up_saved = True

                # Add assistant response to messages for potential next iteration
                messages.append({"role": "assistant", "content": result})

                # Only the LAST iteration's text matters for Telegram
                filtered = self._run_commands_and_extract_message(result, command_processor)

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

            if ran_commands:
                if not follow_up_saved:
                    self._save_autonomous_result_to_history(result)
                result_cleaned = re.sub(r'<thinking>.*?</thinking>', '', result, flags=re.DOTALL)
                result_cleaned = re.sub(r'<[^>]*>', '', result_cleaned)
                result_cleaned = re.sub(r'\[[^\]]*\]', '', result_cleaned).strip()
            else:
                result_cleaned = self.clean_result_and_potentially_add_to_history(result)

            if not re.sub(r'[^A-Za-z]', '', result_cleaned).upper().endswith('NO') and not self.test_mode:
                # Clean filtered output for the Telegram message: drop thinking blocks and
                # bracket markers, but KEEP newlines and Telegram HTML (<b>, <i>, ...) —
                # the Text Chat prompt rules tell the model to format with them, and
                # the chat transport strips any unsupported tags at send time (RichText.sanitize).
                filtered_cleaned = re.sub(r'<thinking>.*?</thinking>', '', filtered, flags=re.DOTALL)
                filtered_cleaned = re.sub(r'\[[^\]]*\]', '', filtered_cleaned).strip()

                if filtered_cleaned and not filtered_cleaned.startswith("NO "):  # one more extra check for flakey LLMs...
                    notifier().send_message(filtered_cleaned)

            return result

    @staticmethod
    def _run_commands_and_extract_message(result: str, command_processor) -> str:
        """Feed the LLM reply through the CommandProcessor as one whitespace-preserving
        chunk (mirroring DialogManager's stream feed) so commands execute while the
        remaining text keeps its newlines and formatting for Telegram. The reply used
        to be .split() into words and re-joined with single spaces, which collapsed
        every newline the model emitted."""
        result_no_thinking = re.sub(r'<thinking>.*?</thinking>', '', result, flags=re.DOTALL)
        filtered = command_processor.process(result_no_thinking.replace("`", ""), None) or ""
        remaining = command_processor.flush_buffer(None)
        if remaining:
            filtered += remaining
        return filtered

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
        if result_cleaned.lower() != "no":
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
        if visible and visible.upper() != "NO":
            parts.append(visible)

        if parts:
            self.prompt_manager.update_history(None, None, "\n".join(parts))

    def schedule(self, minutes: float, reason: str = "") -> None:
        """Wake in *minutes*; *reason* is what the wake is for, carried into the check's prompt
        because the conversation that scheduled it may be trimmed away by then."""
        wake_at = time.time() + minutes * 60
        info(f"Scheduling Autonomous Action Manager to wake in {minutes} minutes"
             + (f" for: {reason}" if reason else " (no reason given)") + "...")
        wakeup = {"at": wake_at, "scheduled_at": time.time(), "reason": reason}
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
            # If the user just interacted (or is interacting), defer briefly so
            # we don't pile a second assistant reply on top of theirs.
            while self.running and self.is_user_interaction_active():
                info("Scheduled wakeup deferred — user interaction active or recent. Retrying shortly.")
                time.sleep(USER_INTERACTION_QUIET_PERIOD_SECONDS)
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
        """The pending wake-ups on disk. A bare timestamp (the file's shape before wake-ups
        carried a reason) is read as a wake-up with no reason, scheduled at an unknown time."""
        try:
            with open(SCHEDULE_FILE, "r") as f:
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        wakeups = []
        for entry in raw:
            if isinstance(entry, dict):
                wakeups.append({"at": entry["at"], "scheduled_at": entry.get("scheduled_at", entry["at"]),
                                "reason": entry.get("reason", "")})
            else:
                wakeups.append({"at": float(entry), "scheduled_at": float(entry), "reason": ""})
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
            info(f"Restoring scheduled wakeup in {remaining:.1f} minutes"
                 + (f" for: {wakeup['reason']}" if wakeup["reason"] else ""))
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
            while self.running and self.is_user_interaction_active():
                info("Missed wakeup deferred — user interaction active or recent. Retrying shortly.")
                time.sleep(USER_INTERACTION_QUIET_PERIOD_SECONDS)
            info(f"Firing missed wakeup {late} late"
                 + (f" for: {wakeup['reason']}" if wakeup["reason"] else ""))
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
        if self.thread.is_alive():
            self.thread.join(timeout=1)
