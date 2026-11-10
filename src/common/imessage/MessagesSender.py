import subprocess
import threading

from common.RateLimiter import RateLimiter
from common.imessage.Handles import is_group_guid
from common.imessage.MessagesHost import SendResult
from common.logging import info, error

SEND_RATE_LIMIT_WINDOW = 60.0
SEND_RATE_LIMIT_MAX = 5
# A conversation is several bubbles a minute, in several chats at once: the sends of a consumer
# that answers in the chat (`send_chat`) have a budget of their own. It still stops a loop — and
# the consumer caps each recipient besides.
CHAT_RATE_LIMIT_MAX = 30
APPLESCRIPT_TIMEOUT = 15
# How long a send waits for the one before it. With its own APPLESCRIPT_TIMEOUT it stays under
# the 40 s a satellite's caller waits for a send (RemoteMessagesHost.READ_TIMEOUTS): a send that
# queued past that would be reported as "may have been sent" and still go out later.
BUSY_SECONDS = 20
APPLESCRIPT_RESULT_IMESSAGE = "sent:imessage"
APPLESCRIPT_RESULT_SMS = "sent:sms"
APPLESCRIPT_RESULT_GROUP = "sent:group"
UNAUTHORIZED_ERROR = "not authorized to control Messages (System Settings → Privacy & Security → Automation)"


class MessagesSender:
    """Sends a text through Messages.app by AppleScript, to a buddy or a group chat.

    A buddy is tried on iMessage first, then on SMS; a group chat is addressed by its guid. The
    sends of the account are rate-limited here, and a refusal of the Apple event — the Automation
    permission missing — is told apart from every other failure. Nothing raises: the outcome is
    a `SendResult`. One send at a time: a satellite serves its callers on several threads, and
    the budget is checked and spent as one step, so two sends at once cannot both take the
    last place in it. A send whose turn has not come within `BUSY_SECONDS` — Messages hanging
    under the ones before it — is not made, and says so."""

    def __init__(self):
        self._rate_limiter = RateLimiter(SEND_RATE_LIMIT_MAX, SEND_RATE_LIMIT_WINDOW)
        self._chat_rate_limiter = RateLimiter(CHAT_RATE_LIMIT_MAX, SEND_RATE_LIMIT_WINDOW)
        self._sending = threading.Lock()    # the limiters are not thread-safe by themselves

    @staticmethod
    def _applescript_quote(s: str) -> str:
        """Escape a string for safe interpolation inside AppleScript double quotes,
        breaking newlines into concatenated `& linefeed &` segments."""
        if s is None:
            return '""'
        parts = s.split("\n")
        escaped = [p.replace("\\", "\\\\").replace('"', '\\"') for p in parts]
        if len(escaped) == 1:
            return f'"{escaped[0]}"'
        return " & linefeed & ".join(f'"{p}"' for p in escaped)

    @classmethod
    def _build_send_applescript(cls, handle: str, body: str) -> str:
        """Build AppleScript that tries iMessage first then falls back to SMS."""
        target = cls._applescript_quote(handle)
        message = cls._applescript_quote(body)
        return f'''
            tell application "Messages"
                set targetBuddy to missing value
                try
                    set iService to 1st service whose service type = iMessage
                    set targetBuddy to buddy {target} of iService
                    send {message} to targetBuddy
                    return "{APPLESCRIPT_RESULT_IMESSAGE}"
                on error
                    try
                        set smsService to 1st service whose service type = SMS
                        set targetBuddy to buddy {target} of smsService
                        send {message} to targetBuddy
                        return "{APPLESCRIPT_RESULT_SMS}"
                    on error errMsg2 number errNum2
                        error errMsg2 number errNum2
                    end try
                end try
            end tell
        '''

    @classmethod
    def _build_group_send_applescript(cls, guid: str, body: str) -> str:
        """Build AppleScript that posts to a group chat. The chat is addressed by its id — the guid
        chat.db stores is the id Messages exposes — and already knows its service, so there is no
        SMS fallback to try."""
        return f'''
            tell application "Messages"
                send {cls._applescript_quote(body)} to chat id {cls._applescript_quote(guid)}
                return "{APPLESCRIPT_RESULT_GROUP}"
            end tell
        '''

    def refusal(self, body: str, chat: bool = False) -> str:
        """Why nothing is sent right now — an empty body, the rate limit — or "" when a send may
        go. `chat`: a send of a conversation, against the chat budget."""
        if not body or not body.strip():
            info("iMessage send skipped: empty body")
            return "empty message body"
        if not self._sending.acquire(blocking=False):
            # a send is under way: this one's budget is asked when its turn comes (`send`)
            return ""
        try:
            return self._over_budget(chat)
        finally:
            self._sending.release()

    def _over_budget(self, chat: bool) -> str:
        """The rate limit's refusal, "" while the budget has room. Called with `_sending` held."""
        rate_err = (self._chat_rate_limiter if chat else self._rate_limiter).check()
        if rate_err:
            info(f"iMessage {'chat ' if chat else ''}send blocked: {rate_err}")
        return rate_err or ""

    @staticmethod
    def chat_target(chat: str) -> str:
        """What a send to a chat addresses: a group by its guid, a 1:1 chat by the buddy its guid
        ends in (`iMessage;-;+46701740605`) — the send that falls back to SMS — and a bare phone
        or email as it is."""
        chat = (chat or "").strip()
        if is_group_guid(chat) or ";" not in chat:
            return chat
        return chat.rsplit(";", 1)[-1]

    def send(self, handle: str, body: str, chat: bool = False) -> SendResult:
        """Run the send for a resolved handle (a phone, an email or a group chat's guid).
        `chat`: counted against the chat budget instead of the account's."""
        if not handle:
            info("iMessage send skipped: nothing to address")
            return SendResult(error="no chat to send to")
        if is_group_guid(handle):
            script = self._build_group_send_applescript(handle, body)
        else:
            script = self._build_send_applescript(handle, body)
        if not self._sending.acquire(timeout=BUSY_SECONDS):
            info(f"iMessage send to {handle} not made: an earlier send still holds Messages after {BUSY_SECONDS}s")
            return SendResult(error=f"Messages is busy with an earlier send ({BUSY_SECONDS} s waited) — this message "
                                    f"was not sent, try again")
        try:
            # asked again here: `refusal` answered before the recipient was resolved, and a send
            # that ran meanwhile may have taken the last place
            rate_err = self._over_budget(chat)
            if rate_err:
                return SendResult(error=rate_err)
            return self._run(script, handle, chat)
        finally:
            self._sending.release()

    def _run(self, script: str, handle: str, chat: bool) -> SendResult:
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=APPLESCRIPT_TIMEOUT
            )
        except subprocess.TimeoutExpired:
            error(f"iMessage send to {handle} timed out after {APPLESCRIPT_TIMEOUT}s")
            return SendResult(error="timeout sending iMessage")
        except Exception as e:
            error(f"iMessage send to {handle} raised: {e}")
            return SendResult(error=f"{e}")

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if result.returncode != 0:
            if "-1743" in stderr or "not authorized" in stderr.lower():
                error(f"iMessage send unauthorized: grant Automation permission to control Messages. stderr={stderr}")
                return SendResult(error=UNAUTHORIZED_ERROR, unauthorized=True)
            error(f"iMessage send to {handle} failed: rc={result.returncode} stderr={stderr}")
            return SendResult(error=stderr or f"osascript exit {result.returncode}")

        (self._chat_rate_limiter if chat else self._rate_limiter).record()
        service = stdout if stdout in (APPLESCRIPT_RESULT_IMESSAGE, APPLESCRIPT_RESULT_SMS,
                                       APPLESCRIPT_RESULT_GROUP) else "sent"
        return SendResult(service=service)
