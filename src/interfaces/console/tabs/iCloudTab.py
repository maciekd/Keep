from typing import List, Optional, Tuple

from interfaces.console import Theme
from interfaces.console.framework.TUIScreen import Color
from interfaces.console.tabs.AuthTab import AuthTab, PENDING
from interfaces.console.widgets.Board import SummaryTile, Table
from assistant.integration.integrations.ICloudIntegration import iCloudAPI, iCloudAuthenticationError
from assistant.integration.IntegrationMonitor import IntegrationMonitor

DEVICE_WIDTH, MODEL_WIDTH, CONFIGURED_WIDTH = 30, 22, 18
NONE_CONFIGURED = "none yet — ask Keep to set one up, or copy an id into icloud.devices"
VERIFIED_STATES = ("authenticated", "partial_auth")   # partial_auth still serves Find My


class iCloudTab(AuthTab):
    """iCloud login with its 2FA, and the Find My devices the session sees."""

    SUBMITTING_MESSAGE = "Verifying 2FA code..."
    STATUS_LABELS = {
        **AuthTab.STATUS_LABELS,
        "partial_auth": ("Partly verified", Theme.COLOR_GREEN),
        "needs_2fa": ("2FA Required", Theme.COLOR_YELLOW),
    }

    def __init__(self):
        super().__init__()
        self._icloud_devices: list = []
        self._devices_list: str = ""

    def _connect(self):
        return iCloudAPI.get_instance()

    def _on_status(self, api):
        if self.status not in VERIFIED_STATES:
            return
        try:
            self._icloud_devices = api.get_icloud_devices() or []
        except Exception:
            self._icloud_devices = []
        try:
            self._devices_list = api.get_devices_list() or ""
        except Exception:
            self._devices_list = ""

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
        """Every device the account sees: a green dot and the name it answers to in voice
        commands, Find My pings and location detection when icloud.devices lists it."""
        rows = []
        if self.status in VERIFIED_STATES:
            for device in self._icloud_devices:
                configured = device.get("configured_as")
                rows.append((Theme.COLOR_GREEN if configured else Theme.COLOR_FAINT, [
                    (device.get("name", "Unknown"), Theme.COLOR_TEXT),
                    (device.get("model") or PENDING, Theme.COLOR_LABEL),
                    (configured, Theme.COLOR_TEAL) if configured else (PENDING, Theme.COLOR_FAINT),
                    (str(device.get("id", "")), Theme.COLOR_DIM)]))
        return Table("Find My devices",
                     [("Device", DEVICE_WIDTH), ("Model", MODEL_WIDTH), ("Configured as", CONFIGURED_WIDTH), ("Id", None)],
                     rows, "sign in to see the devices")

    def _tiles(self) -> List[SummaryTile]:
        devices = self._icloud_devices if self.status in VERIFIED_STATES else []
        seen: SummaryTile = ("Devices", str(len(devices)), Theme.COLOR_BRIGHT, [("seen by the account", Theme.COLOR_LABEL)])
        if not devices:
            seen = ("Devices", PENDING, Theme.COLOR_DIM, [("sign in to see them", Theme.COLOR_DIM)])
        if not self._devices_list:
            return [seen, ("Configured", "none", Theme.COLOR_DIM, [(NONE_CONFIGURED, Theme.COLOR_DIM)])]
        configured = sum(1 for device in devices if device.get("configured_as"))
        return [seen, ("Configured", str(configured) if devices else "", Theme.COLOR_TEAL,
                       [(self._devices_list, Theme.COLOR_LABEL)])]

    def _entry_hint(self) -> Optional[Tuple[str, Color]]:
        if self.status == "partial_auth":
            return "Auto-2FA on standby — Ctrl-E to verify with an SMS code now (optional)", Theme.COLOR_GREEN
        if self.status == "needs_2fa":
            return "Auto-2FA in progress — Ctrl-E to enter the code manually", Theme.COLOR_YELLOW
        return None

    def _begin_entry(self):
        if self.status in ("needs_2fa", "partial_auth"):
            self._request_sms_and_enter_code()

    def _request_sms_and_enter_code(self):
        """Request an SMS code (unless one is already on its way — a second request would
        invalidate it), then take the code by hand."""
        api = self._get_api()
        if not api:
            self._status_message = "Error: API not initialized"
            return
        if api.sms_request_pending():
            self._status_message = "SMS code already requested — auto-submit armed; type it to enter manually"
            super()._begin_entry()
            return
        try:
            phone = api.request_sms_2fa()
            self._status_message = f"SMS code sent to {phone}"
            super()._begin_entry()
        except Exception as e:
            self._status_message = f"SMS request failed: {e}"

    def _verify_code(self, api, code: str):
        if api.send_2fa_code(code):
            self._status_message = "2FA verification successful!"
            IntegrationMonitor.get_instance().check_now("iCloud")
            self._check_auth_status()
        else:
            self._status_message = "2FA verification failed. Try again."

    def _refresh(self):
        """Refresh the iCloud session; synchronous, like the check that follows it."""
        self._status_message = "Refreshing connection..."
        api = self._get_api()
        if api:
            try:
                api.refresh_connection()
                self._status_message = "Connection refreshed"
            except iCloudAuthenticationError as e:
                self._status_message = f"Auth error: {e}"
            except Exception as e:
                self._status_message = f"Error: {e}"
        self._check_auth_status()
