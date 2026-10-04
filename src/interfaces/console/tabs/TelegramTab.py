from typing import List, Optional, Tuple

from interfaces.console import Theme
from interfaces.console.framework.TUIScreen import Color
from interfaces.console.tabs.AuthTab import (AuthTab, BACKSPACE_KEYS, ENTER_KEYS, PENDING, STANDARD_ENTRY_HINT,
                                             code_display)
from interfaces.console.widgets import Board
from interfaces.console.widgets.Board import Fact, SummaryTile
from common.logging import trace

INPUT_PHONE = "phone"
INPUT_CODE = "code"
INPUT_PASSWORD = "password"

STEPS = (INPUT_PHONE, INPUT_CODE, INPUT_PASSWORD)

CODE_LENGTH = 5
PHONE_MAX = 16
PHONE_MIN = 8
MASK = "•"
CURSOR = "▏"

# The Feed tile's value colour per FeedState value; any other state is dim
FEED_COLORS = {"live": Theme.COLOR_GREEN, "catching_up": Theme.COLOR_YELLOW, "starting": Theme.COLOR_YELLOW, "failed": Theme.COLOR_RED}


class TelegramTab(AuthTab):
    """The Telegram feed's login: phone number, the code Telegram sends, and the account's
    2FA password when it has one."""

    HELP = "Ctrl-R reconnect  Ctrl-P authenticate"
    STATUS_LABELS = {
        **AuthTab.STATUS_LABELS,
        "needs_auth": ("Not authenticated", Theme.COLOR_YELLOW),
        "connecting": ("Connecting", Theme.COLOR_YELLOW),
    }

    def __init__(self):
        super().__init__()
        self._phone_number = ""
        self._feed: Optional[dict] = None

    def _connect(self):
        from assistant.integration.integrations.TelegramFeedIntegration import TelegramFeedIntegration
        return TelegramFeedIntegration.get_instance()

    def _on_status(self, api):
        from assistant.integration.FeedRunner import Feeds
        try:
            self._feed = Feeds.get_instance().status(api.feed_name)
        except Exception as e:
            trace(f"TelegramTab: no feed status: {e}")
            self._feed = None

    def _tiles(self) -> List[SummaryTile]:
        feed = self._feed
        if not feed:
            return [("Feed", PENDING, Theme.COLOR_DIM, [("not running", Theme.COLOR_DIM)])]
        state = feed["state"]
        segments = [(f"{feed['stored']:,} {feed['items']} stored", Theme.COLOR_LABEL)]
        if feed.get("remaining"):
            segments.append((f"{feed['remaining']:,} to go", Theme.COLOR_DIM))
        if feed.get("last_error"):
            segments = [(str(feed["last_error"]), Theme.COLOR_RED)]
        return [("Feed", "paused" if feed.get("paused") else state.replace("_", " "),
                 FEED_COLORS.get(state, Theme.COLOR_DIM), segments)]

    def _fields(self) -> List[Fact]:
        """The login's three steps: the one being typed bright under a yellow label, the
        answered ones above it, the ones to come faint. The password is never drawn."""
        mode, text = self._input_mode, self._input_text
        if mode not in STEPS:
            hint = self._entry_hint()
            return [Fact("Phone", hint[0], hint[1]) if hint else Fact("Phone", PENDING, Theme.COLOR_FAINT),
                    Fact("Code", PENDING, Theme.COLOR_FAINT),
                    Fact("Password", PENDING, Theme.COLOR_FAINT)]
        step = STEPS.index(mode)
        return Board.steps(step, [
            ("Phone", text + CURSOR if step == 0 else self._phone_number),
            ("Code", code_display(text, CODE_LENGTH) if step == 1 else "sent" if step > 1 else PENDING),
            ("Password", MASK * len(text) + CURSOR if step == 2 else PENDING)])

    def _hints(self) -> str:
        if self._input_mode == INPUT_PHONE:
            return f"With the country code, e.g. +48123456789  {STANDARD_ENTRY_HINT}"
        return super()._hints()

    def _entry_hint(self) -> Optional[Tuple[str, Color]]:
        if self.status == "needs_auth":
            return "Ctrl-P starts the login", Theme.COLOR_YELLOW
        return None

    def _handle_key(self, key: str) -> bool:
        if key == '\x10':  # Ctrl+P
            self._input_mode = INPUT_PHONE
            self._input_text = "+"
            self._status_message = ""
            return True
        return False

    def _handle_entry_key(self, key: str) -> bool:
        mode, text = self._input_mode, self._input_text
        if key == 'escape':
            self._cancel_login()
        elif key in ENTER_KEYS:
            trace(f"TelegramTab: Enter pressed in {mode} mode, buffer='{text}' (len={len(text)})")
            if mode == INPUT_PHONE and len(text) >= PHONE_MIN:
                self._submit_phone()
            elif mode == INPUT_CODE and len(text) >= CODE_LENGTH:
                self._submit_code()
            elif mode == INPUT_PASSWORD and text:
                self._submit_password()
            else:
                trace("TelegramTab: Enter ignored — input too short")
        elif key in BACKSPACE_KEYS:
            if text and (mode != INPUT_PHONE or len(text) > 1):   # the phone keeps its '+'
                self._input_text = text[:-1]
        elif mode == INPUT_PHONE:
            if key.isdigit() and len(text) < PHONE_MAX:
                self._input_text += key
        elif mode == INPUT_CODE:
            if key.isdigit() and len(text) < CODE_LENGTH:
                self._input_text += key
                if len(self._input_text) == CODE_LENGTH:
                    self._submit_code()
        elif mode == INPUT_PASSWORD:
            if len(key) == 1 and ord(key) >= 32:
                self._input_text += key
        return True

    def _cancel_login(self):
        """Esc: the entry ends and the integration drops its login client, so an unfinished
        login leaves nothing behind."""
        self._cancel_entry()
        api = self._get_api()
        if api:
            trace("TelegramTab: login cancelled")
            self._run(api.abort_login)

    def _submit_phone(self):
        phone = self._input_text
        self._phone_number = phone
        trace(f"TelegramTab: sending code request to {phone}")

        def send(api):
            sent = api.send_code_request(phone)
            trace(f"TelegramTab: code sent successfully to {phone}")
            self._status_message = f"Code sent {api.code_destination(sent)} — waiting for it."
            self._input_mode = INPUT_CODE
            self._input_text = ""

        self._act("Sending verification code...", send)

    def _submit_code(self):
        code, phone = self._input_text, self._phone_number

        def sign_in(api):
            try:
                api.sign_in(phone, code)
            except Exception as e:
                err_msg = str(e).lower()
                if "password" in err_msg or "2fa" in err_msg or "two-step" in err_msg:
                    self._status_message = "2FA password required"
                    self._input_mode = INPUT_PASSWORD
                    self._input_text = ""
                    return
                raise
            self._signed_in(api)

        self._act("Signing in...", sign_in)

    def _submit_password(self):
        password = self._input_text

        def sign_in(api):
            api.sign_in_2fa(password)
            self._signed_in(api)

        self._act("Verifying 2FA password...", sign_in)

    def _signed_in(self, api):
        """The sign-in wrote the session file; the restart builds the instance that opens it."""
        self._status_message = "Authentication successful!"
        api.restart_feed()
        self._check_auth_status()

    def _refresh(self):
        self._status_message = "Reconnecting..."

        def reconnect():
            api = self._get_api()
            if api:
                try:
                    api.restart_feed()
                    self._status_message = "Reconnect initiated"
                except Exception as e:
                    self._status_message = f"Error: {e}"
            self._check_auth_status()

        self._run(reconnect)
