import time
from typing import List, Optional, Tuple

from interfaces.console import Theme
from interfaces.console.framework.TUIScreen import Color
from interfaces.console.tabs.AuthTab import AuthTab, PENDING
from interfaces.console.widgets.Board import SummaryTile, Table
from assistant.integration.integrations.cameras.ArloCameraIntegration import ArloCamera, ArloAuthenticationError
from assistant.integration.IntegrationMonitor import IntegrationMonitor

VERIFY_POLL_SECONDS = 30   # pyaarlo processes the code on its own thread; poll for the login


class ArloTab(AuthTab):
    """Arlo login with its 2FA, and the cameras the account has."""

    def __init__(self):
        super().__init__()
        self._camera_names: list = []
        self._rate_limit_until: float = 0

    def _connect(self):
        return ArloCamera.get_instance()

    def _on_status(self, api):
        if self.status == "authenticated":
            try:
                self._camera_names = api.get_camera_names()
            except Exception:
                self._camera_names = []

    def _table(self) -> Table:
        names = self._camera_names if self.status == "authenticated" else []
        return Table("Cameras", [("Camera", None)],
                     [(Theme.COLOR_GREEN, [(name, Theme.COLOR_TEXT)]) for name in names], "connect to see the cameras")

    def _tiles(self) -> List[SummaryTile]:
        count = len(self._camera_names) if self.status == "authenticated" else 0
        cameras: SummaryTile = ("Cameras", str(count), Theme.COLOR_BRIGHT, [("on the account", Theme.COLOR_LABEL)])
        if not count:
            cameras = ("Cameras", PENDING, Theme.COLOR_DIM, [("connect to see them", Theme.COLOR_DIM)])
        remaining = self._rate_limit_remaining()
        if remaining <= 0:
            return [cameras]
        mins, secs = divmod(int(remaining), 60)
        return [cameras, ("Rate limited", f"{mins:02d}:{secs:02d}", Theme.COLOR_ORANGE,
                          [("until the next attempt is allowed", Theme.COLOR_LABEL)])]

    def _rate_limit_remaining(self) -> float:
        return self._rate_limit_until - time.time() if self._rate_limit_until > 0 else 0.0

    def _entry_hint(self) -> Optional[Tuple[str, Color]]:
        if self.status == "error" and "2FA" in self._auth_status.get("message", "").upper():
            return "Press Ctrl+E to enter 2FA code", Theme.COLOR_YELLOW
        return None

    def _outcome(self) -> str:
        """While rate limited, a countdown to the next allowed attempt replaces the outcome."""
        if self._rate_limit_until > 0:
            remaining = self._rate_limit_until - time.time()
            if remaining > 0:
                mins, secs = divmod(int(remaining), 60)
                return f"Rate limited — next attempt allowed in {mins:02d}:{secs:02d}"
            self._rate_limit_until = 0
            self._status_message = ""
        return self._status_message

    def _verify_code(self, api, code: str):
        api.tfa_handler.submit_code(code)
        self._status_message = "Verifying 2FA code..."
        for i in range(VERIFY_POLL_SECONDS):
            time.sleep(1)
            self._check_auth_status()
            if self.status == "authenticated":
                break
            self._status_message = f"Verifying 2FA code ({i + 1}s)..."
        if self.status == "authenticated":
            self._status_message = "2FA verification successful!"
            IntegrationMonitor.get_instance().check_now("Arlo")
        else:
            self._status_message = "2FA code submitted, waiting for authentication..."

    def _refresh(self):
        api = self._get_api()
        if api:
            cooldown = api.auth_cooldown_remaining
            if cooldown > 0:
                self._rate_limit_until = time.time() + cooldown
                return

        self._rate_limit_until = 0
        self._status_message = "Connecting..."

        def refresh():
            if api:
                try:
                    api.refresh_connection()
                    self._status_message = "Connection refreshed"
                except ArloAuthenticationError as e:
                    cooldown = api.auth_cooldown_remaining
                    if cooldown > 0:
                        self._rate_limit_until = time.time() + cooldown
                    else:
                        self._status_message = f"Auth error: {e}"
                except Exception as e:
                    self._status_message = f"Error: {e}"
            self._check_auth_status()

        self._run(refresh)
