"""MessagesSender: the AppleScript it builds, and what each way osascript can end means."""
import subprocess
import threading
from unittest.mock import Mock, patch

import pytest

from common.imessage import MessagesSender as module
from common.imessage.MessagesSender import CHAT_RATE_LIMIT_MAX, MessagesSender, SEND_RATE_LIMIT_MAX


def osascript(returncode=0, stdout="sent:imessage", stderr=""):
    return Mock(returncode=returncode, stdout=stdout, stderr=stderr)


def run(**result):
    return patch.object(module.subprocess, "run", return_value=osascript(**result))


@pytest.mark.unit
class TestApplescriptQuote:
    def test_double_quote_escaped(self):
        assert MessagesSender._applescript_quote('say "hi"') == '"say \\"hi\\""'

    def test_backslash_escaped(self):
        assert MessagesSender._applescript_quote('a\\b') == '"a\\\\b"'

    def test_newline_split_into_linefeed_concat(self):
        assert MessagesSender._applescript_quote("line1\nline2") == '"line1" & linefeed & "line2"'


@pytest.mark.unit
class TestSend:
    def test_a_buddy_send_runs_osascript_with_the_handle_and_the_body(self):
        with run() as mock_run:
            result = MessagesSender().send("+12345678901", "hello there")
        assert (result.error, result.service) == ("", "sent:imessage")
        command = mock_run.call_args[0][0]
        assert command[0] == "osascript"
        assert '"+12345678901"' in command[2] and '"hello there"' in command[2]
        assert "buddy" in command[2] and "chat id" not in command[2]

    def test_a_group_send_addresses_the_chat_by_its_guid(self):
        with run(stdout="sent:group") as mock_run:
            result = MessagesSender().send("any;+;chat100", "dinner at 7")
        script = mock_run.call_args[0][0][2]
        assert 'chat id "any;+;chat100"' in script and "buddy" not in script
        assert '"dinner at 7"' in script
        assert result.service == "sent:group"

    def test_body_with_quotes_and_newlines_escaped(self):
        with run() as mock_run:
            MessagesSender().send("+12345678901", 'She said "hi"\nSee you')
        script = mock_run.call_args[0][0][2]
        assert '\\"hi\\"' in script
        assert "linefeed" in script

    def test_an_answer_messages_did_not_script_is_still_a_send(self):
        with run(stdout="something else"):
            result = MessagesSender().send("+12345678901", "hi")
        assert (result.error, result.service) == ("", "sent")

    def test_a_refused_apple_event_is_the_missing_automation_permission(self):
        with run(returncode=1, stdout="",
                 stderr="execution error: Not authorized to send Apple events to Messages. (-1743)"):
            result = MessagesSender().send("+12345678901", "hi")
        assert result.unauthorized
        assert "not authorized" in result.error.lower() and "Automation" in result.error

    def test_generic_failure_returns_stderr(self):
        with run(returncode=1, stdout="", stderr="something exploded"):
            result = MessagesSender().send("+12345678901", "hi")
        assert result.error == "something exploded" and not result.unauthorized

    def test_a_failure_without_stderr_names_the_exit_code(self):
        with run(returncode=3, stdout="", stderr=""):
            assert "3" in MessagesSender().send("+12345678901", "hi").error

    def test_timeout_returns_timeout_error(self):
        with patch.object(module.subprocess, "run", side_effect=subprocess.TimeoutExpired(cmd="osascript", timeout=15)):
            result = MessagesSender().send("+12345678901", "hi")
        assert "timeout" in result.error

    def test_a_missing_osascript_is_an_error_value_not_an_exception(self):
        with patch.object(module.subprocess, "run", side_effect=FileNotFoundError("osascript")):
            assert "osascript" in MessagesSender().send("+12345678901", "hi").error


@pytest.mark.unit
class TestRefusal:
    def test_an_empty_body_is_refused(self):
        assert MessagesSender().refusal("  \n") == "empty message body"

    def test_a_body_may_go_while_the_account_is_under_its_rate(self):
        assert MessagesSender().refusal("hi") == ""

    def test_rate_limit_blocks_after_threshold(self):
        sender = MessagesSender()
        with run():
            for _ in range(SEND_RATE_LIMIT_MAX):
                assert sender.refusal("ping") == ""
                assert sender.send("+12345678901", "ping").error == ""
        assert "rate limit" in sender.refusal("ping")

    @pytest.mark.parametrize("chat, budget", [(False, SEND_RATE_LIMIT_MAX), (True, CHAT_RATE_LIMIT_MAX)],
                             ids=["the account's", "the chat's"])
    def test_two_sends_at_once_cannot_both_take_the_last_place_in_the_budget(self, chat, budget):
        """A satellite serves its callers on several threads: both asked `refusal` while one
        place was left, and both went out."""
        sender = MessagesSender()
        together = threading.Barrier(2)
        reached = []

        def messages(*args, **kwargs):
            reached.append(args)
            try:
                together.wait(0.03)     # two sends inside osascript at once, when nothing keeps them apart
            except threading.BrokenBarrierError:
                pass
            return osascript()

        with run():
            for _ in range(budget - 1):
                sender.send("+46701740605", "ping", chat=chat)
        answers = []

        def send():
            if not sender.refusal("ping", chat=chat):
                answers.append(sender.send("+46701740605", "ping", chat=chat))

        with patch.object(module.subprocess, "run", side_effect=messages):
            threads = [threading.Thread(target=send) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
        assert len(reached) == 1, "one message reached Messages"
        assert sorted(bool(answer.error) for answer in answers) in ([False], [False, True])
        assert "rate limit" in sender.refusal("ping", chat=chat)

    def test_a_send_that_cannot_get_its_turn_is_not_made_and_says_so(self):
        """Messages hanging under an earlier send: queued past the 40 s its caller waits, this
        one would be reported as "may have been sent" and still go out afterwards."""
        sender = MessagesSender()
        sender._sending.acquire()       # an earlier send is still inside osascript
        with run() as messages, patch.object(module, "BUSY_SECONDS", 0.01):
            assert sender.refusal("ping") == "", "the budget is asked when the send's turn comes, not waited for"
            result = sender.send("+46701740605", "ping")
        messages.assert_not_called()
        assert result.error and not result.service
        sender._sending.release()
        with run() as messages:
            assert sender.send("+46701740605", "ping").error == "", "the turn it did not get is not kept from the next"
        messages.assert_called_once()

    def test_the_wait_for_a_turn_and_the_send_itself_fit_in_what_a_satellites_caller_waits(self):
        """A contract with RemoteMessagesHost's read timeout for a send."""
        from common.imessage.RemoteMessagesHost import READ_TIMEOUTS
        assert module.BUSY_SECONDS + module.APPLESCRIPT_TIMEOUT < min(READ_TIMEOUTS["send"], READ_TIMEOUTS["send_chat"])

    def test_a_failed_send_does_not_count_against_the_rate(self):
        sender = MessagesSender()
        with run(returncode=1, stdout="", stderr="something exploded"):
            for _ in range(SEND_RATE_LIMIT_MAX + 1):
                sender.send("+12345678901", "ping")
        assert sender.refusal("ping") == ""


@pytest.mark.unit
class TestChatSends:
    """What a consumer that answers in the chat sends: addressed as the chat is, on a budget of its own."""

    @pytest.mark.parametrize("chat, target", [
        ("any;+;chat2", "any;+;chat2"), ("iMessage;+;chat9", "iMessage;+;chat9"),
        ("any;-;+46701740605", "+46701740605"), ("iMessage;-;m@mlab.test", "m@mlab.test"),
        ("SMS;-;+46701740605", "+46701740605"), ("+46701740605", "+46701740605"), (" m@mlab.test ", "m@mlab.test"),
    ], ids=["a group", "a group, older spelling", "a direct chat", "a direct chat by email", "an SMS chat",
            "a bare phone", "a bare email"])
    def test_a_chat_is_addressed_by_its_guid_only_when_it_is_a_group(self, chat, target):
        assert MessagesSender.chat_target(chat) == target

    def test_a_send_to_nothing_never_reaches_messages(self):
        with run() as osa:
            assert MessagesSender().send("", "hi", chat=True).error
        osa.assert_not_called()

    def test_the_two_budgets_do_not_draw_on_each_other(self):
        sender = MessagesSender()
        with run():
            for _ in range(SEND_RATE_LIMIT_MAX):
                sender.send("+46701740605", "ping")
            assert "rate limit" in sender.refusal("ping")
            assert sender.refusal("ping", chat=True) == ""
            for _ in range(CHAT_RATE_LIMIT_MAX):
                assert sender.send("+46701740605", "ping", chat=True).error == ""
        assert "rate limit" in sender.refusal("ping", chat=True)
