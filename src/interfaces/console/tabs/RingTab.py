from typing import List, Optional, Tuple

from interfaces.console import Theme
from interfaces.console.framework.TUIScreen import Color
from interfaces.console.tabs.AuthTab import AuthTab, PENDING
from interfaces.console.widgets.Board import SummaryTile, Table
from assistant.integration.integrations.cameras.RingIntegration import RingIntegration, RingAuthenticationError
from assistant.integration.IntegrationMonitor import IntegrationMonitor


class RingTab(AuthTab):
    """Ring login with its 2FA, and the doorbells the account has."""

    def __init__(self):
        super().__init__()
        self._doorbell_names: list = []

    def _connect(self):
        return RingIntegration.get_instance()

    def _on_status(self, api):
        if self.status == "authenticated":
            try:
                self._doorbell_names = api.get_doorbell_names()
            except Exception:
                self._doorbell_names = []

    def _before_draw(self):
        """The integration verifies codes from iMessage on its own; show the outcome when it has one."""
        api = self._get_api()
        auto_msg = api.auto_2fa_message if api else None
        if auto_msg:
            self._status_message = auto_msg
            self._input_mode = False
            self._input_text = ""
            self._check_auth_status_async()

    def _table(self) -> Table:
        names = self._doorbell_names if self.status == "authenticated" else []
        return Table("Doorbells", [("Doorbell", None)],
                     [(Theme.COLOR_GREEN, [(name, Theme.COLOR_TEXT)]) for name in names], "connect to see the doorbells")

    def _tiles(self) -> List[SummaryTile]:
        count = len(self._doorbell_names) if self.status == "authenticated" else 0
        if not count:
            return [("Doorbells", PENDING, Theme.COLOR_DIM, [("connect to see them", Theme.COLOR_DIM)])]
        return [("Doorbells", str(count), Theme.COLOR_BRIGHT, [("on the account", Theme.COLOR_LABEL)])]

    def _entry_hint(self) -> Optional[Tuple[str, Color]]:
        if self.status == "error" and "2FA" in self._auth_status.get("message", "").upper():
            return "Press Ctrl+E to enter 2FA code", Theme.COLOR_YELLOW
        return None

    def _verify_code(self, api, code: str):
        api.tfa_handler.submit_code(code)
        self._status_message = "Verifying 2FA code..."
        self._check_auth_status()
        if self.status == "authenticated":
            self._status_message = "2FA verification successful!"
            IntegrationMonitor.get_instance().check_now("Ring")
        else:
            self._status_message = "2FA code submitted, waiting for authentication..."

    def _refresh(self):
        self._status_message = "Connecting..."

        def refresh():
            api = self._get_api()
            if api:
                try:
                    api.refresh_connection()
                    self._status_message = "Connection refreshed"
                except RingAuthenticationError as e:
                    self._status_message = f"Auth error: {e}"
                except Exception as e:
                    self._status_message = f"Error: {e}"
            self._check_auth_status()

        self._run(refresh)
