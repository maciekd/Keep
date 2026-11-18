import os
import re
import uuid
import queue
import threading
import time

from common.DevelopmentConfig import LOCAL_DEVELOPMENT_MODE
from .formats.CommandFormats import CommandFormats
from .formats.RescueFormat import RescueFormat
from .CommandEcho import describe_completion
from .Commands import Commands
from common.logging import error, trace, info, note, fine, masked_for_log
from assistant.AssistantEvents import CommandRan, Thinking, ThinkingDone, observers
from assistant.ChannelProfile import AUTONOMOUS, ChannelProfile, profile_for
from assistant.Notifier import notifier


# kill() queues this so the worker thread wakes from command_queue.get() and exits instead
# of blocking there for the life of the process (one such thread per LLM round, before).
_STOP = (None, None, None, None)


class CommandProcessor:
    """Filters the LLM token stream, stripping <thinking> blocks and executing embedded commands.

    Built fresh for every LLM round; commands run on a worker thread and their replies are
    fed back to the dialog for the next round.
    """
    DEBUG = False
    THINKING_TAG_START = "<thinking>"
    THINKING_TAG_END = "</thinking>"
    THINKING_TAG_START_SPACED = "< thinking >"
    # The close of a thinking block the model never opened in the stream (DeepSeek's own
    # tag, written after a call when the server did not route the block as reasoning).
    BARE_THINK_END = "</think>"
    UNTERMINATED_COMMAND_SHOWN_CHARS = 300
    # The runaway guard: a reply is read up to this many calls and the stream is dropped
    # there (2026-10-21: Opus wrote 51 history searches in one reply, a 660 KB result).
    # The calls read run as usual; the stored turn is cut after the last of them
    # (`CommandFormat.cut`), so the next round looks like a reply of exactly that many — and
    # the command reply says the rest was dropped, so a follow-up round always comes
    # (a cap of silent commands alone would end the request with the next one lost unmentioned).
    # 15 since 2026-10-23: the largest honest reply seen on prod in the cap's first two days ran
    # nine, and the two runaways it caught (the same call repeated) hit any cap within a second.
    MAX_COMMANDS_PER_REPLY = 15
    CUT_NOTE = ("Your reply was cut after {cap} commands, the most one reply may run: the first {cap} "
                "ran, nothing after the last of them was read, so any later command in the reply did "
                "not run and no text after it reached the user. Continue from here, {cap} commands per "
                "reply at most.")
    # The fabricated-results guard: a model that drifts into its trained tool loop writes the
    # tool's reply itself — `<function_results>` holding an invented toolkit, invented search
    # hits, an invented "Sent 2 photos." — and goes on as if it had read them (prod 2026-10-25,
    # Opus on a photo question: the whole block reached the Telegram chat as reply text, and
    # a declaration line inside the invented toolkit ran as a call). Only the system writes
    # results: the stream is cut where the tag opens, the calls before it ran as usual, their
    # real results come back in the follow-up round, and the stored turn ends at the last of
    # them (`kept_of`), so the history never shows the model its own invented results.
    FABRICATED_RESULTS_OPEN = "<function_results"
    # The after-the-fetch guard (issue #134): once a reply has dispatched a stop-and-wait
    # command (Commands.is_stop_and_wait — the toolkits' sections and the three loads), its
    # result does not exist yet, so anything the reply goes on to write rests on nothing — a
    # verdict posted on a PR, "✅ created" for a save that failed, a price list the model wrote
    # for the page it had just opened (prod, 25–26 Oct 2026). Further stop-and-wait calls
    # still pass (several fetches in one reply are common and fine); the first non-whitespace
    # text or any other call cuts the reply there, as the cap does: not run, not shown, the
    # stream closed, the stored turn ending at the last fetch, and the note below ahead of
    # the real result in the follow-up round. A call written before the fetch has run already.
    FETCH_CUT_NOTE = ("Your reply went on after `{fetch}` before its result existed, so it was cut there: "
                      "{dropped} dropped unread, and nothing after that command reached the user. The result "
                      "follows — answer from it now, and issue again any call you still want.")
    FABRICATED_RESULTS_NOTE = (
        "Your reply wrote a <function_results> block of its own{after}. Only the system writes "
        "results, so that block and everything after it were dropped unread and did not reach the "
        "user. {real}")
    # The repeat guard (#123): a reply gets no result between its calls, so the same call
    # twice in a row is one result twice — never for a stop-and-wait command (prod 2026-11-04:
    # `load_watch` 15 times to an "ok", a 129 KB result), and for any other command only as
    # far as its effect adds up (`volume_up` ×3 is a louder room; ×15 is a runaway). The
    # reply is cut at the repeat that passes the allowance, as the cap cuts: the duplicate
    # does not run, nothing after it is read, and the stored turn ends at the last call that
    # ran — cut rather than skipped, since `kept_of` keeps the first `commands_dispatched`
    # calls of the raw text, which only holds while every call read so far was dispatched.
    MAX_IDENTICAL_CALLS = 3
    REPEAT_CUT_NOTE = ("Your reply wrote the call {label} {times} times in a row. A reply gets no result "
                       "between its calls, so {why}: the reply was cut at copy {times}, {ran} command(s) "
                       "ran, the rest was dropped unread and no text after it reached the user. The "
                       "result(s) follow.")
    # The degenerate-value guard (#123, prod 2026-10-29): a stream that drifts into bare
    # `<invoke>` blocks whose every value is quotes and brackets alone (`"`, `}`, `]`) — 313
    # `browse_eval` blocks after one real call, 14 of them run as JavaScript. Only a rescued
    # call is checked (every occurrence came through the rescue; a call in the taught syntax
    # never looks like this), a call with no values is left alone, and the reply is cut at the
    # first such call like the repeat above.
    DEGENERATE_VALUE_RE = re.compile(r'^[\s"\'`{}\[\](),]*$')
    DEGENERATE_CUT_NOTE = ("Your reply wrote a `{name}` call (as an <invoke> block) whose value(s) held nothing "
                           "but punctuation: {values}. That is no call, so the reply was cut there: {ran} "
                           "command(s) ran, the rest was dropped unread and no text after it reached the "
                           "user. The result(s) follow.")

    def __init__(self, test_suite_mode: bool):
        self.previous_content_buffer = ""
        self.thinking_buffer = ""
        self.thinking_content_buffer = ""
        self.in_thinking_tag = False
        self.thinking_start_time = None
        self.test_mode_dont_actually_execute_commands = test_suite_mode
        self.command_running_in_background = False
        self.command_being_processed = False
        # command_reply is composed from two sources and never edited in place: what the
        # synchronous path appended this round (errors, denials, the unterminated-command
        # report) and the background batch's results as last published. A publish carries
        # every result of the round, so it replaces the batch part whole — composing from
        # the parts is what keeps a result from going to the LLM twice when a later batch
        # publishes after an error was appended.
        self.command_reply = None
        self.command_reply_contains_error = False
        self._sync_replies: list[str] = []
        # What the request processor tells the model about the round itself (its reply was
        # cut at the output cap): in front of the results, no error framing, and kept when
        # a background batch publishes after it.
        self._notes: list[str] = []
        self._batch_reply: str | list | None = None
        self._batch_reply_contains_error = False
        # The texts of the batch's results one by one, as they stand in command_reply: the
        # history's cap is shared between them.
        self.result_parts: list[str] = []
        # Written on the command worker thread; read by _handle_background_commands after
        # the queue drains, so no lock is needed. A fresh CommandProcessor is built per
        # LLM round, which keeps both round-scoped.
        self.executed_commands: list[str] = []
        # Set by flush_buffer on the request thread, kept apart from command_reply: a command
        # still running from the same reply replaces command_reply wholesale when it finishes.
        self.unterminated_command_report: str | None = None
        self.commands_dispatched = 0
        # Set when the cap above is reached: the rest of the stream is ignored, and the
        # dialog closes it.
        self.reply_cut = False
        # The stop-and-wait command this reply dispatched last, None before the first: with
        # one pending, text and other calls cut the reply (FETCH_CUT_NOTE).
        self.fetch_pending: str | None = None
        # The reply was cut right after its last call (the after-the-fetch, repeat and
        # degenerate-value guards): the stored turn ends at that call, never at prose after it.
        self.cut_after_last_call = False
        # The call this reply dispatched last, as (name, params), and how many times in a row
        # it was written — the repeat guard's memory.
        self._last_call: tuple[str, tuple[str, ...]] | None = None
        self._last_call_times = 0
        # The post-call hold (AbstractLLM.resumes_thinking_after_calls): the text written
        # after the last dispatched call, kept back until a bare </think> says it was
        # reasoning — dropped, and nothing in it is parsed as a call — or the stream ends and
        # it was the reply. None while no hold is on.
        self._post_call_hold: str | None = None
        # The spans dropped this round, as written, for `without_dropped_thinking`.
        self.dropped_thinking: list[str] = []
        # When the turn this processor serves began, for a processor without a request
        # handler (the autonomous check sets it); see Commands.turn_started.
        self.turn_started: float | None = None
        self.commands = Commands(self)
        self.possible_commands = self.commands.list_commands()
        # The active format, with Claude's <invoke> block read beside it for the call a model
        # writes in that shape anyway (RescueFormat); the stored turn is rewritten to the
        # active syntax by `format.rewritten` before the dialog keeps it.
        self.format = RescueFormat.wrap(CommandFormats.active()(self.possible_commands))
        self.wait_has_been_done_for_this_request = False
        self.is_killed = False
        # For the orphaned-command handoff (see kill()): the most recently dispatched
        # command's display string, and the registry entry to resolve once a command
        # that outlived this (killed) processor publishes its reply.
        self.last_dispatched_display: str | None = None
        self._orphan_entry = None

        self._streaming = True

        self.command_queue = queue.Queue()
        self._worker_stop_sent = False
        self.worker_thread = threading.Thread(target=self._command_worker, daemon=True)
        self.worker_thread.start()

    def debug(self, msg: str):
        if self.DEBUG:
            trace("[DEBUG] " + msg.replace("\n", " "))

    # Runs on: command worker thread (daemon, started in __init__).
    # Picks commands from queue, optionally sleeps to not interrupt the user, then executes.
    # Synchronous commands (turn_on, dim) run inline. Background commands (google, open_url)
    # are dispatched to Commands.background_command() which spawns a bg command worker thread.
    def _command_worker(self):
        # Runs until kill()'s stop sentinel arrives — never leaving on is_killed alone, so every
        # queued item gets its task_done() and a command_queue.join() after a kill cannot hang.
        while True:
            command_name, params, dialog_manager, unreadable = self.command_queue.get()
            if command_name is None:
                self.command_queue.task_done()
                trace("Command worker: killed — thread exiting.")
                break
            if self.is_killed:
                trace(f"Command worker: dropping queued {command_name} — killed.")
                self.command_queue.task_done()
                continue
            self.command_being_processed = True
            task_done_called = False
            try:
                if dialog_manager is not None and dialog_manager.is_killed:
                    self.command_queue.task_done()
                    task_done_called = True
                    continue

                if not self.wait_has_been_done_for_this_request and dialog_manager is not None:
                    if not dialog_manager.web_request_handler.wait_so_that_response_timing_is_natural(0):
                        self.command_queue.task_done()
                        task_done_called = True
                        continue
                    self.wait_has_been_done_for_this_request = True

                params_display = ", ".join(params) if isinstance(params, list) else str(params)
                self.debug(f"Cleaned parameters: {params_display}")
                self.last_dispatched_display = self._label(command_name, params_display)

                blocked = False
                if unreadable:
                    self.append_command_reply(unreadable)
                    blocked = True
                elif getattr(self._context_for(dialog_manager), 'guest', False) is True:
                    info(f"Command {command_name} refused: the request is from a guest, every command is "
                         f"disabled for the turn")
                    self.append_command_reply(
                        f"Command {self._label(command_name, params_display)} refused: the request is not from one of "
                        f"the household's owners, so every command is disabled for this turn.")
                    blocked = True
                elif (not self.test_mode_dont_actually_execute_commands
                        and self.commands.is_sensitive(command_name)):
                    blocked = self._gate_sensitive_command(command_name, params, dialog_manager)

                if blocked:
                    pass  # error already appended; skip execution and the echo
                elif command_name == "false_positive" and LOCAL_DEVELOPMENT_MODE:
                    trace(f"Do run FP commands in LOCAL_DEVELOPMENT_MODE mode.")
                    self._echo_command(command_name, params, params_display, dialog_manager)
                    self.commands.run(command_name, params)
                elif self.test_mode_dont_actually_execute_commands and not self.commands.is_pure(command_name):
                    note(f"Command {self._label(command_name, params_display)} (test mode, won't execute any commands)")
                else:
                    self.commands._current_profile = self._profile_for(dialog_manager)
                    self.commands._current_context = self._context_for(dialog_manager)
                    self.commands.turn_started = self._turn_started_for(dialog_manager)
                    result = self._run_echoed(command_name, params, params_display, dialog_manager)
                    if result:
                        error_msg = f"Command {self._label(command_name, params_display)} returned an error: \"{result}\"."
                        note(error_msg)
                        self.append_command_reply(error_msg)
                    else:
                        self.executed_commands.append(f"{command_name}({params_display})")

                if command_name:
                    observers().on_command_ran(command_name, params_display)
                    self._emit(dialog_manager, CommandRan(command_name, params_display))
            finally:
                self.command_being_processed = False
                if not task_done_called:
                    self.command_queue.task_done()

    # Runs on: command worker thread. Echoes the command to the user's chat as it starts
    # and runs it; the echo is completed here for a command that ran inline, and by the
    # background worker for one that Commands handed to it (see Commands.background_command).
    def _run_echoed(self, command_name: str, params, params_display: str, dialog_manager):
        echo = self._echo_command(command_name, params, params_display, dialog_manager)
        self.commands._current_echo = echo
        started = time.time()
        result = None
        failed = True
        try:
            result = self.commands.run(command_name, params)
            failed = bool(result)
        finally:
            if echo is not None and self.commands._current_echo is echo:
                self.commands._current_echo = None
                self._complete_echo(echo, describe_completion(started, failed=failed))
        return result

    @staticmethod
    def _complete_echo(echo, detail: str):
        try:
            notifier().complete_command_notification(echo, detail)
        except Exception as e:
            error(f"Failed to complete command notification: {e}")

    # The channel profile decides whether the user is reading (echo the command to the
    # chat) or talking (VoiceToVoice: no echo); autonomous runs have no user. Returns the
    # notifier's handle for the echo, or None when nothing was shown.
    def _echo_command(self, command_name: str, params, params_display: str, dialog_manager):
        profile = self._profile_for(dialog_manager)
        if dialog_manager is None:
            trace(f"Skipping the chat echo for autonomous command {command_name}({params_display})")
            return None
        if not profile.command_notifications:
            trace(f"Skipping the chat echo for {profile.name} command {command_name}({params_display})")
            return None
        try:
            text = self.commands.echo_text(command_name, params)
            if not text:
                return None
            ctx = self._context_for(dialog_manager)
            handle = notifier().send_command_notification(
                text, ephemeral=profile.notification_ephemeral,
                source=ctx.source, conversation=ctx.conversation)
            if profile.notify_observers_of_commands:
                observers().on_command_notification(text)
            return handle
        except Exception as e:
            error(f"Failed to send command notification: {e}")
            return None

    # Runs on: command worker thread. Blocks until the user approves/denies via the chat
    # buttons — asked in the chat the request came from, so another person's chat never
    # sees it (or the request times out / approval channel is unavailable). On non-approve
    # outcomes, appends an explanatory error to command_reply so the LLM sees it, and
    # returns True so the worker skips both execution and the post-run notification.
    def _gate_sensitive_command(self, command_name: str, params: list[str], dialog_manager=None) -> bool:
        from .Commands import normalize_llm_escapes

        params_display = ", ".join(params) if isinstance(params, list) else str(params)
        # Run the preview through the same escape-sequence conversion the executors
        # apply (cmd_send_imessage / cmd_send_email), so the approval dialog shows
        # real line breaks rather than literal "\n" — preview must match delivery.
        params_display = normalize_llm_escapes(params_display)
        title = self.commands.describe(command_name, params) or self._label(command_name, params_display)
        details = params_display or self.commands.approval_note(command_name)
        info(f"Sensitive command '{command_name}' awaiting user approval: {title}")
        ctx = self._context_for(dialog_manager)
        decision = notifier().request_approval(
            title=title, details=details,
            source=getattr(ctx, 'source', None), conversation=getattr(ctx, 'conversation', None),
            cancelled=lambda: self._killed(dialog_manager))

        if decision == "approve" and self._killed(dialog_manager):
            # Approved into a dead request (the kill landed between the answer and this check):
            # the user superseded the turn, so the command they approved must not run.
            decision = "cancelled"
        if decision == "approve":
            info(f"Sensitive command '{command_name}' approved.")
            return False
        if decision == "cancelled":
            info(f"Sensitive command '{command_name}' not run: the request was killed while its approval was pending.")
            return True

        label = self._label(command_name, params_display)
        msg_by_decision = {
            "deny": f"Command {label} was denied by user.",
            "timeout": f"Command {label} approval timed out — not executed.",
            "unavailable": f"Command {label} blocked: approval channel (notifier) is unavailable.",
        }
        error_msg = msg_by_decision.get(decision, f"Command '{command_name}' approval returned: {decision}")
        note(error_msg)
        self.append_command_reply(error_msg)
        return True

    @property
    def parser(self):
        """The format under its old name, for a caller that still asks for the parser."""
        return self.format

    def is_possibly_command(self, content: str) -> bool:
        return self.format.is_possibly_command(content)

    # Runs on: a background lane thread. `content` is the batch's results — all of the
    # round's so far — and `contains_error` whether one of them failed.
    def set_command_reply(self, content: str | list, contains_error: bool = False,
                          parts: list[str] | None = None):
        info("Command done.")
        self._batch_reply = content
        self.result_parts = parts or []
        self._batch_reply_contains_error = contains_error
        self._compose_reply()
        self._log_command_reply(content)
        self._resolve_orphan_if_any()

    # Only called with command errors, approval denials, the unterminated-command report and
    # the cut note — flags the reply so _handle_background_commands frames the follow-up round
    # as an error report.
    def append_command_reply(self, content: str):
        self._sync_replies.append(content)
        self._compose_reply()
        self._log_command_reply(content)
        self._resolve_orphan_if_any()

    def add_note(self, content: str):
        """A note about the round for the model, ahead of the results — the round circles
        back for it even when no command ran."""
        self._notes.append(content)
        self._compose_reply()
        self._log_command_reply(content)

    def _compose_reply(self):
        """The notes and the synchronous parts first, one per line, then the batch: in front
        of a block list as one text block, before a text reply with a blank line between."""
        sync_text = "\n".join(self._notes + self._sync_replies)
        batch = self._batch_reply
        if isinstance(batch, list):
            self.command_reply = ([{"type": "text", "text": sync_text}] if sync_text else []) + batch
        else:
            self.command_reply = "\n\n".join(part for part in (sync_text, batch) if part) or None
        self.command_reply_contains_error = bool(self._sync_replies) or self._batch_reply_contains_error

    def clear_command_reply(self):
        """The reply was consumed: the next one starts from nothing, the batch results
        included — kept, the next publish would carry this reply's results again."""
        self.command_reply = None
        self.command_reply_contains_error = False
        self._sync_replies = []
        self._notes = []
        self._batch_reply = None
        self._batch_reply_contains_error = False
        self.result_parts = []
        self.commands.forget_batch_results()

    def _log_command_reply(self, content: str | list):
        from assistant.RequestProcessor import _save_image_to_log
        from common.logging import disable_file_logging
        if disable_file_logging:
            return
        command_reply_path = "log/command_replies"
        if not os.path.exists(command_reply_path):
            os.makedirs(command_reply_path)
        reply_id = str(uuid.uuid4())
        fn = f"{command_reply_path}/{reply_id}.txt"
        if isinstance(content, list):
            parts = []
            for i, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "image":
                    path = _save_image_to_log(block, f"cmd_{reply_id[:8]}", i)
                    parts.append(f"[IMAGE: {path}]" if path else "[IMAGE]")
                elif isinstance(block, dict):
                    parts.append(block.get("text", str(block)))
                else:
                    parts.append(str(block))
            log_text = "\n".join(parts)
        else:
            log_text = content
        with open(fn, "w") as file:
            file.write(masked_for_log(log_text))
            trace(f"Saving command reply to {fn}.")

    # Runs on: request thread (during LLM stream processing). Queues work for command worker thread.
    def invoke_command(self, command_name: str, params: list[str], dialog_manager):
        self.commands_dispatched += 1
        self.command_queue.put((command_name, params, dialog_manager, None))

    def _answer_unreadable(self, command_name: str, params: list[str], dialog_manager, answer: str):
        """A call that must not run (`_unreadable`) takes its place in the queue like any
        call, so its answer stands among the results in the order the reply was written."""
        self.commands_dispatched += 1
        self.command_queue.put((command_name, params, dialog_manager, answer))

    # A value is the text between its quotes. A call that goes on after a closing quote is
    # code around the value — prod 2026-11-17: a mail body written as a triple-quoted block,
    # `.replace(…) if False else` and a second block ran as one body holding both blocks and
    # the code between them, and went to an outside company. After a block nothing may follow,
    # whatever the command; after a plain string the text has always been glued to the value
    # (a remembered sentence with quotes of its own), which a command behind the approval gate
    # — a message somebody receives — is too costly to run on.
    UNREADABLE_NOTE = ("{label} was not run: the call goes on after the closing quote of a value "
                       "({stray}), so the value could not be read as it was meant. A value is the final text "
                       "itself, whole inside its quotes — never an expression around it, and a quote inside a "
                       "plain string is escaped or the text put in a triple-quoted block. Issue the call again.")
    UNREADABLE_SHOWN_CHARS = 80

    def _unreadable(self, command_name: str, display: str, content: str) -> str | None:
        """Why the call just read must not run, as the answer for the model, or None."""
        stray = self.format.text_after_quote(content, blocks_only=not self.commands.is_sensitive(command_name))
        if not stray:
            return None
        note(f"{command_name} not run: the call goes on after a closing quote ({stray[:self.UNREADABLE_SHOWN_CHARS]!r})")
        label = self._label(command_name, display)
        if len(label) > self.UNTERMINATED_COMMAND_SHOWN_CHARS:
            label = label[:self.UNTERMINATED_COMMAND_SHOWN_CHARS] + "…"
        shown = stray if len(stray) <= self.UNREADABLE_SHOWN_CHARS else stray[:self.UNREADABLE_SHOWN_CHARS] + "…"
        return self.UNREADABLE_NOTE.format(label=label, stray=shown)

    def reset_parse_state(self):
        """Forget partial parse progress from an aborted stream so a transparent retry
        starts clean (see DialogManager._stream_with_transport_retry). Only safe when
        no command has been dispatched from the aborted stream."""
        self.previous_content_buffer = ""
        self.thinking_buffer = ""
        self.thinking_content_buffer = ""
        self.in_thinking_tag = False
        self.thinking_start_time = None

    def start_new_reply(self):
        """A later reply fed through the same processor (the autonomous check keeps one for
        all its rounds; a request builds one per round): the parse state and the cap are
        the reply's, so they start over. Left standing, a cut in one reply dropped every
        later reply of the check unread (2026-10-22)."""
        if self.commands_dispatched or self.reply_cut:
            trace(f"New reply through the same processor: the cap starts over "
                  f"({self.commands_dispatched} dispatched, cut: {self.reply_cut})")
        self.reset_parse_state()
        self.commands_dispatched = 0
        self.reply_cut = False
        self.fetch_pending = None
        self.cut_after_last_call = False
        self._last_call = None
        self._last_call_times = 0
        self.unterminated_command_report = None
        if hasattr(self.format, "rescued"):
            self.format.rescued = 0   # the rescue counts per reply, and logs so

    # Runs on: command worker thread (called from Commands.background_command)
    def background_command_start(self):
        trace("Running command in background.")
        self.command_running_in_background = True

    # Runs on: a background lane thread (called from Commands._finish_one when the
    # batch's last command is done)
    def background_command_done(self):
        self.command_running_in_background = False

    def process_one_command(self, content: str, dialog_manager) -> str:
        rescued_before = getattr(self.format, "rescued", 0)
        call, remainder = self.format.extract(content)
        if call is not None:
            command_name, params, display = call
            rescued = getattr(self.format, "rescued", 0) > rescued_before
            if rescued and self._cut_at_degenerate_values(command_name, params):
                return ""
            if self._cut_at_repeat(command_name, params, display):
                return ""
            if self.fetch_pending and not self.commands.is_stop_and_wait(command_name):
                self._cut_after_fetch(f"the call {self._label(command_name, display)}")
                return ""
            if unreadable := self._unreadable(command_name, display, content):
                self._answer_unreadable(command_name, params, dialog_manager, unreadable)
                return remainder
            self.invoke_command(command_name, params, dialog_manager)
            if self.commands.is_stop_and_wait(command_name):
                self.fetch_pending = command_name
            elif self._streaming and self._resumes_thinking(dialog_manager):
                # Text after a stop-and-wait call cuts the reply unshown anyway (_released).
                trace(f"Holding the text after {command_name} until a bare </think> or the end of the stream")
                self._post_call_hold = ""
            return remainder
        if remainder is not None:
            info(f"A complete call block without a call in it — dropped, the reply goes on after it: {content[:80]!r}")
            return remainder
        error("Could not find command, will truncate response.")
        return ""

    def _is_partial_tag_match(self, content: str, tag: str) -> int:
        """
        Check if content ends with a partial tag.
        Returns the length of the partial match, or 0 if no match.
        Also checks for spaced variant '< thinking >' when checking for '<thinking>'.
        """
        for length in range(min(len(tag) - 1, len(content)), 0, -1):
            if content.endswith(tag[:length]):
                return length
        if tag == self.THINKING_TAG_START:
            # Min length 3: skip "< " (length 2) which is too ambiguous and can
            # consume spaces after HTML tags like <b>, <code> etc. when the '<'
            # from a tag ends up in thinking_buffer and the next token is a space.
            # Length 1 ("<") is already handled by the non-spaced check above.
            for length in range(min(len(self.THINKING_TAG_START_SPACED) - 1, len(content)), 2, -1):
                if content.endswith(self.THINKING_TAG_START_SPACED[:length]):
                    return length
        return 0

    @staticmethod
    def _context_for(dialog_manager):
        """The RequestContext of the dialog being processed; None for autonomous runs and
        handlers without one."""
        if dialog_manager is None:
            return None
        try:
            return dialog_manager.web_request_handler.ctx
        except AttributeError:
            return None

    def _turn_started_for(self, dialog_manager) -> float | None:
        """When the turn being processed began: the request's arrival for a dialog with a
        handler, else what the processor's owner set (the autonomous check sets its own)."""
        handler = getattr(dialog_manager, "web_request_handler", None)
        started = getattr(handler, "turn_started", None)
        return self.turn_started if started is None else started

    @classmethod
    def _profile_for(cls, dialog_manager) -> ChannelProfile:
        """The channel profile of the request being processed. No dialog (autonomous runs)
        or a handler without a request context → AUTONOMOUS."""
        ctx = cls._context_for(dialog_manager)
        return AUTONOMOUS if ctx is None else profile_for(ctx.source)

    @staticmethod
    def _emit(dialog_manager, event):
        """Put an event on the request's stream; dialogs without a stream (autonomous runs,
        bare test doubles) simply drop it."""
        emit = getattr(dialog_manager, "emit", None) if dialog_manager is not None else None
        if callable(emit):
            emit(event)

    @staticmethod
    def _output_token_estimate(dialog_manager) -> int:
        try:
            return int(dialog_manager.current_output_token_estimate())
        except (AttributeError, TypeError, ValueError):
            return 0

    def _emit_thinking_chunk(self, chunk: str, dialog_manager):
        """A thinking chunk always goes to the request's own stream; it reaches the observer
        hub only when the channel profile forwards it (TextChat)."""
        if not chunk:
            return
        self._emit(dialog_manager, Thinking(chunk, self._output_token_estimate(dialog_manager)))
        if self._profile_for(dialog_manager).forward_thinking:
            observers().on_thinking_chunk(chunk)

    def _finalize_thinking(self, duration: float, dialog_manager):
        """The current thinking block is complete."""
        self._emit(dialog_manager, ThinkingDone(duration))
        if self._profile_for(dialog_manager).forward_thinking:
            observers().on_thinking_done(duration)

    def _filter_thinking_tags(self, content: str, dialog_manager=None) -> tuple[str, str]:
        """
        Filter out <thinking>...</thinking> tags from content.
        Returns (filtered_content, buffered_partial).
        buffered_partial contains text that might be part of a tag boundary.

        Also streams thinking content to the observer hub (TextChat source only)
        as it accumulates, so a dashboard can show the model think in real time.
        """
        result = ""
        i = 0

        while i < len(content):
            if self.in_thinking_tag:
                end_pos = content.find(self.THINKING_TAG_END, i)
                if end_pos != -1:
                    chunk = content[i:end_pos]
                    self.thinking_content_buffer += chunk
                    self._emit_thinking_chunk(chunk, dialog_manager)
                    thinking_duration = time.time() - self.thinking_start_time
                    if thinking_duration > 0:
                        fine(f"AI thought for {thinking_duration:.2f}s:")
                    fine(self.thinking_content_buffer.strip())
                    self._finalize_thinking(thinking_duration, dialog_manager)
                    self.thinking_content_buffer = ""
                    self.in_thinking_tag = False
                    self.thinking_start_time = None
                    i = end_pos + len(self.THINKING_TAG_END)
                else:
                    # Check if content ends with partial </thinking>
                    remaining = content[i:]
                    partial_len = self._is_partial_tag_match(remaining, self.THINKING_TAG_END)
                    if partial_len > 0:
                        chunk = remaining[:-partial_len]
                        self.thinking_content_buffer += chunk
                        self._emit_thinking_chunk(chunk, dialog_manager)
                        return result, remaining[-partial_len:]
                    self.thinking_content_buffer += remaining
                    self._emit_thinking_chunk(remaining, dialog_manager)
                    return result, ""
            else:
                start_pos = content.find(self.THINKING_TAG_START, i)
                start_pos_spaced = content.find(self.THINKING_TAG_START_SPACED, i)
                # Use whichever tag is found first
                if start_pos_spaced != -1 and (start_pos == -1 or start_pos_spaced < start_pos):
                    start_pos = start_pos_spaced
                    tag_len = len(self.THINKING_TAG_START_SPACED)
                else:
                    tag_len = len(self.THINKING_TAG_START)
                if start_pos != -1:
                    result += content[i:start_pos]
                    self.in_thinking_tag = True
                    self.thinking_start_time = time.time()
                    # Cover the silent thinking gap with a short spoken filler.
                    if dialog_manager is not None:
                        dialog_manager.notify_thinking_started()
                    i = start_pos + tag_len
                else:
                    remaining = content[i:]
                    partial_len = self._is_partial_tag_match(remaining, self.THINKING_TAG_START)
                    if partial_len > 0:
                        result += remaining[:-partial_len]
                        return result, remaining[-partial_len:]
                    result += remaining
                    return result, ""

        return result, ""

    # Runs on: request thread (called from DialogManager.process_llm_stream during LLM token streaming)
    def process(self, content: str, dialog_manager) -> str | None:
        if self.reply_cut:
            return None
        content = self.thinking_buffer + content
        self.thinking_buffer = ""

        filtered_content, partial_tag = self._filter_thinking_tags(content, dialog_manager)
        self.thinking_buffer = partial_tag

        if self.in_thinking_tag or partial_tag:
            if filtered_content:
                return self._process_commands(filtered_content, dialog_manager)
            return None

        return self._process_commands(filtered_content, dialog_manager)

    def _process_commands(self, content: str, dialog_manager) -> str | None:
        if self._post_call_hold is not None:
            content = self._through_hold(content)
            if content is None:
                return None
        self.previous_content_buffer += content

        self.debug(f"content: '{content}' previous_content_buffer: '{self.previous_content_buffer}'")

        if (at := self.previous_content_buffer.find(self.FABRICATED_RESULTS_OPEN)) >= 0:
            # The calls before the tag run as usual; the tag and all after it are dropped.
            head, dropped = self.previous_content_buffer[:at], self.previous_content_buffer[at:]
            self.previous_content_buffer = ""
            shown = self._process_commands(head, dialog_manager) if head else None
            if not self.reply_cut:
                self._cut_at_fabricated_results(dropped)
            return shown

        if self.is_possibly_command(self.previous_content_buffer):
            self.debug(f"  (possibly command)")
            if self.format.complete(self.previous_content_buffer, streaming=self._streaming):
                self.debug(f"Processing command!")
                self.previous_content_buffer = self.process_one_command(self.previous_content_buffer, dialog_manager)
                if self.commands_dispatched >= self.MAX_COMMANDS_PER_REPLY:
                    info(f"Reply cut after {self.commands_dispatched} commands — the rest of the stream is "
                         f"dropped unread: {self.previous_content_buffer[:80]!r}")
                    self.reply_cut = True
                    self.previous_content_buffer = ""
                    self.append_command_reply(self.CUT_NOTE.format(cap=self.MAX_COMMANDS_PER_REPLY))
                    return None
                # Go again on what is left. It was filtered for thinking tags before it entered
                # the buffer, so it goes straight back here: `process` would prepend the partial
                # tag the filter is holding, text that arrived after this remainder, not before it.
                self.debug(f"rest: '{self.previous_content_buffer}'")
                p = self.previous_content_buffer
                self.previous_content_buffer = ""
                return self._process_commands(p, dialog_manager)
            return None
        else:
            # Scan for a command embedded later in the buffer
            cmd_pos = self.format.find_start(self.previous_content_buffer)
            if cmd_pos > 0:
                self.debug(f"found embedded command at pos {cmd_pos}")
                before = self.previous_content_buffer[:cmd_pos]
                self.previous_content_buffer = self.previous_content_buffer[cmd_pos:]
                if (before := self._released(before)) is None:
                    return None
                rest = self._process_commands("", dialog_manager)
                return before + rest if rest else before

            # Before flushing, check if the buffer tail could be the start of a command name.
            # This prevents losing partial command names when they arrive in the same token as preceding text.
            if self._streaming:
                tail_pos = self._partial_tail(self.previous_content_buffer)
                if tail_pos == 0:
                    self.debug("the whole buffer may open a results block — retaining it")
                    return None
                if tail_pos > 0:
                    self.debug(f"retaining partial command tail at pos {tail_pos}")
                    before = self.previous_content_buffer[:tail_pos]
                    self.previous_content_buffer = self.previous_content_buffer[tail_pos:]
                    return self._released(before)
            elif (tail_pos := self._results_tail(self.previous_content_buffer)) >= 0 \
                    and len(self.previous_content_buffer) - tail_pos >= 3:
                info(f"Stream ended on the start of a <function_results> tag — dropping it: "
                     f"{self.previous_content_buffer[tail_pos:]!r}")
                self.previous_content_buffer = self.previous_content_buffer[:tail_pos]

            self.debug(f"no command possible")
            p = self.previous_content_buffer
            self.previous_content_buffer = ""
            return self._released(p)

    @staticmethod
    def _resumes_thinking(dialog_manager) -> bool:
        ai = getattr(dialog_manager, "ai", None)
        resumes = getattr(ai, "resumes_thinking_after_calls", None)
        return callable(resumes) and resumes() is True

    def _through_hold(self, content: str) -> str | None:
        """`content` added to the post-call hold: None while the hold goes on, or the text
        that follows a bare </think> in it — the hold up to the tag was the model's reasoning
        and is dropped, logged and remembered for the stored turn — with the hold over."""
        self._post_call_hold += content
        at = self._post_call_hold.find(self.BARE_THINK_END)
        if at < 0:
            return None
        end = at + len(self.BARE_THINK_END)
        held, self._post_call_hold = self._post_call_hold, None
        # The whitespace between the call and the block stays: it separates the call from
        # what follows, in the reply and in the stored turn.
        lead = len(held) - len(held.lstrip())
        dropped, rest = held[lead:end], held[:lead] + held[end:]
        self.dropped_thinking.append(dropped)
        note(f"A second thinking block after a call, closed with a bare </think>, dropped from the reply "
             f"({len(dropped)} chars): {dropped[:120]!r}")
        return rest

    def _end_hold(self) -> str:
        """The stream ended with the post-call hold on: what it holds is the reply — unless it
        closes with the bare tag after all, in which case it was reasoning and is dropped."""
        held, self._post_call_hold = self._post_call_hold, None
        if (at := held.find(self.BARE_THINK_END)) >= 0:
            lead = len(held) - len(held.lstrip())
            end = at + len(self.BARE_THINK_END)
            self.dropped_thinking.append(held[lead:end])
            note(f"The reply ended in a second thinking block after a call — dropped ({at - lead} chars before the tag)")
            return held[:lead] + held[end:]
        if held.strip():
            trace(f"The text held after the last call was the reply — released whole: {len(held)} chars")
        return held

    def without_dropped_thinking(self, raw: str) -> str:
        """`raw` (the reply as received) without the thinking blocks the hold dropped, so the
        stored turn never teaches the model the slip."""
        for span in self.dropped_thinking:
            if span in raw:
                raw = raw.replace(span, "", 1)
            else:
                trace(f"A dropped thinking block is not in the raw reply as written ({len(span)} chars) — left")
        return raw

    def _released(self, text: str) -> str | None:
        """`text` as the reply may show it: itself, or None when a stop-and-wait command is
        pending and the text says anything — the reply is cut there (FETCH_CUT_NOTE)."""
        if self.fetch_pending and text.strip():
            self._cut_after_fetch(f"{len(text.strip())} chars of text")
            return None
        return text

    def _cut_at_repeat(self, command_name: str, params: list[str], display: str) -> bool:
        """The repeat guard: True when this call is the one too many of a run of identical
        calls — the reply is cut there and the call does not run."""
        call = (command_name, tuple(params))
        if call != self._last_call:
            self._last_call, self._last_call_times = call, 1
            return False
        self._last_call_times += 1
        times = self._last_call_times
        stop_and_wait = self.commands.is_stop_and_wait(command_name)
        if not stop_and_wait and times <= self.MAX_IDENTICAL_CALLS:
            trace(f"The call {command_name} repeated: {times} in a row, within the allowance of {self.MAX_IDENTICAL_CALLS}")
            return False
        why = (f"the same {command_name} twice has one result" if stop_and_wait
               else f"a reply runs one call at most {self.MAX_IDENTICAL_CALLS} times in a row")
        label = self._label(command_name, display)
        allowed = 1 if stop_and_wait else self.MAX_IDENTICAL_CALLS
        note(f"Reply cut at the repeated call {label}: written {times} times in a row, {allowed} allowed "
             f"— the repeat does not run, the rest is dropped unread")
        self._cut_reply(self.REPEAT_CUT_NOTE.format(label=label, times=times, why=why, ran=self.commands_dispatched))
        return True

    def _cut_at_degenerate_values(self, command_name: str, params: list[str]) -> bool:
        """The degenerate-value guard: True when the (rescued) call has values and every one
        of them is punctuation alone — the reply is cut there and the call does not run."""
        if not params or not all(self.DEGENERATE_VALUE_RE.match(value) for value in params):
            return False
        values = ", ".join(repr(value) for value in params)
        note(f"Reply cut at a rescued {command_name} call whose every value is punctuation alone ({values}) "
             f"— it does not run, the rest is dropped unread")
        self._cut_reply(self.DEGENERATE_CUT_NOTE.format(name=command_name, values=values, ran=self.commands_dispatched))
        return True

    def _cut_reply(self, note_text: str):
        """Stop reading the reply here, with the stored turn ending at the last call that ran
        and the note ahead of the results in the follow-up round, which comes for the note
        even when no call ran."""
        self.reply_cut = True
        self.cut_after_last_call = True
        self.previous_content_buffer = ""
        self.add_note(note_text)

    def _cut_after_fetch(self, dropped: str):
        """The reply went on after a stop-and-wait command: stop reading it here, run and show
        nothing more of it, and say so ahead of the fetch's result in the follow-up round."""
        info(f"Reply cut after the stop-and-wait command {self.fetch_pending}: {dropped} written before its "
             f"result existed — dropped unread")
        self._cut_reply(self.FETCH_CUT_NOTE.format(fetch=self.fetch_pending, dropped=dropped))

    def _partial_tail(self, buffer: str) -> int:
        """Where a possible start of a call or of a `<function_results` tag begins at the end
        of the buffer: text to hold until the next chunk says what it is; -1 when none."""
        tails = [pos for pos in (self.format.partial_tail(buffer), self._results_tail(buffer)) if pos >= 0]
        return min(tails) if tails else -1

    @classmethod
    def _results_tail(cls, buffer: str) -> int:
        """Where the buffer ends in a proper prefix of the `<function_results` tag, -1 when
        it does not (`<` alone counts: the invoke rescue holds it the same way)."""
        tag = cls.FABRICATED_RESULTS_OPEN
        for length in range(min(len(tag) - 1, len(buffer)), 0, -1):
            if buffer.endswith(tag[:length]):
                return len(buffer) - length
        return -1

    def _cut_at_fabricated_results(self, dropped: str):
        """The model wrote a results block itself: stop reading the reply here and tell it so
        in the follow-up round, which comes for this note even when no call ran."""
        ran = self.commands_dispatched
        error(f"The reply wrote a <function_results> block of its own after {ran} command(s) — the stream "
              f"is dropped from the tag on: {dropped[:160]!r}")
        self.reply_cut = True
        after = f" after {ran} command(s)" if ran else " before any command"
        real = (f"The real result(s) of the {ran} command(s) that ran follow." if ran
                else "No command ran, so there is no result.")
        self.append_command_reply(self.FABRICATED_RESULTS_NOTE.format(after=after, real=real))

    def kept_of(self, raw: str) -> str:
        """What the dialog stores of a reply it stopped reading (`reply_cut`): the text up to
        and including the last call that ran, every block closed, and never a results block
        the model wrote — so the follow-up round reads a reply of exactly the calls that ran."""
        kept = self.format.cut(raw, self.commands_dispatched, after_last=self.cut_after_last_call)
        if (at := kept.find(self.FABRICATED_RESULTS_OPEN)) >= 0:
            kept = kept[:at]
        return kept

    def flush_buffer(self, dialog_manager) -> str | None:
        """Flush any remaining buffer content with strict (non-streaming) parsing.
        Called after the LLM stream ends so commands like set_volume(level: 5)
        that were held back during streaming get processed."""
        if not self.previous_content_buffer and not self.thinking_buffer and self._post_call_hold is None:
            return None
        self._streaming = False
        try:
            held = self._end_hold() if self._post_call_hold is not None else ""
            flushed = self._process_commands(held, dialog_manager)
        finally:
            self._streaming = True
        if flushed is None and self.previous_content_buffer:
            leftover = self.previous_content_buffer
            self.previous_content_buffer = ""
            if not leftover.strip():
                trace(f"Stream ended with whitespace after the last command — dropping it: {leftover!r}")
            elif command_name := self.format.unterminated(leftover):
                self._report_unterminated_command(command_name, leftover, dialog_manager)
            elif self._MALFORMED_TAG_RE.match(leftover):
                self._report_malformed_tag(leftover, dialog_manager)
            else:
                # Held back only because it could still have grown into `name(` — it never did.
                info(f"Stream ended on a bare command name, no call — releasing it as reply text: {leftover!r}")
                flushed = self._released(leftover)
        return flushed

    @staticmethod
    def _label(command_name: str, params_display: str) -> str:
        """How a command is named in the texts that go back to the model and the approval
        card: `name (params)` — no call syntax, since the model's own may be any of the
        command formats."""
        return f"{command_name} ({params_display})" if params_display else command_name

    def _report_unterminated_command(self, command_name: str, leftover: str, dialog_manager):
        error(f"Stream ended inside an unterminated {command_name}( call — not run: {leftover!r}")
        self._report_not_run(f"{command_name} was not run: the reply ended inside the call before it was "
                             f"closed, so the call could not be read.", leftover, dialog_manager)

    # Text the parser held back as a possible call that turned out to open with a tag and hold
    # another one: a call in a shape no syntax reads (prod 2026-10-22: `<invoke_load_toolkit>`,
    # before the fused tag was taught). A lone tag followed by prose is not one.
    _MALFORMED_TAG_RE = re.compile(r"\s*<[A-Za-z/][^<>]*>.*<", re.S)

    def _report_malformed_tag(self, leftover: str, dialog_manager):
        error(f"Stream ended on tag-shaped text that is no call in any syntax read here — not run: {leftover!r}")
        self._report_not_run("Nothing was run: the reply ended in tag-shaped text that is not a call in the "
                             "syntax this prompt teaches, so no call could be read from it.", leftover, dialog_manager)

    def _report_not_run(self, what: str, leftover: str, dialog_manager):
        """Queue `what` for the circle-back round, with the dropped text quoted."""
        if dialog_manager is None:
            info("No dialog to circle back through (autonomous run) — the dropped call is only logged.")
            return
        if self.is_killed or dialog_manager.is_killed:
            info("Request was killed mid-command — the dropped call is not reported to the LLM.")
            return
        shown = leftover.strip()
        if len(shown) > self.UNTERMINATED_COMMAND_SHOWN_CHARS:
            shown = shown[:self.UNTERMINATED_COMMAND_SHOWN_CHARS] + "…"
        self.unterminated_command_report = (
            f"{what} Nothing after that point in the reply reached the user either. Received: {shown}")

    def _killed(self, dialog_manager) -> bool:
        return self.is_killed or (dialog_manager is not None and dialog_manager.is_killed)

    def take_unterminated_command_report(self) -> str | None:
        report, self.unterminated_command_report = self.unterminated_command_report, None
        return report

    def kill(self):
        self.is_killed = True
        self._stop_worker()
        self._register_orphaned_bg_command()

    def _stop_worker(self):
        """Wake the worker out of command_queue.get() so its thread ends. Idempotent, and unlike
        kill() it hands nothing to the orphan registry — for teardown paths."""
        if not self._worker_stop_sent:
            self._worker_stop_sent = True
            self.command_queue.put(_STOP)

    # --- orphaned background command handoff --------------------------------------------
    # A kill (request superseded by a new message) does not stop an in-flight background
    # command; its reply used to be silently dropped, leaving the next request's LLM
    # guessing about what happened. Register the situation so the next request can be told.

    def _register_orphaned_bg_command(self):
        from .OrphanedCommands import orphan_registry
        desc = self.last_dispatched_display or "a background command"
        # Background commands of one reply run in parallel; the batch publishes as one
        # reply, so it hands off as one entry naming how many are still running.
        in_flight = self.commands.background_outstanding()
        if in_flight > 1:
            desc = f"{in_flight} background commands (the last one dispatched: {desc})"
        if self.command_reply is not None:
            info(f"Killed with an unconsumed command reply from {desc} — preserving it for the next request.")
            orphan_registry().register_completed(
                desc, self._reply_as_text(self.command_reply), self.command_reply_contains_error)
        elif self.command_running_in_background:
            info(f"Killed while {desc} is still running — its outcome will reach the next request.")
            self._orphan_entry = orphan_registry().register_pending(desc)
        else:
            trace("Killed with no background command in flight — nothing to hand off.")

    # Runs on: bg command worker thread. Publishes the late reply of a command whose
    # request was killed before it finished.
    def _resolve_orphan_if_any(self):
        if self._orphan_entry is None:
            return
        from .OrphanedCommands import orphan_registry
        orphan_registry().resolve(self._orphan_entry, self._reply_as_text(self.command_reply),
                                  self.command_reply_contains_error)
        self._orphan_entry = None

    @staticmethod
    def _reply_as_text(content: str | list | None) -> str:
        if content is None:
            return ""
        if isinstance(content, list):
            parts = [block.get("text", "[image]") if isinstance(block, dict) else str(block)
                     for block in content]
            return "\n".join(parts)
        return str(content)
