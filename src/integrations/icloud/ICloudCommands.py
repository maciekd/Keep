from assistant.SystemNote import system_note
from assistant.commands.CommandSet import CommandSet
from common.logging import error, info
from integrations.icloud.ICloudIntegration import iCloudAPI, iCloudAPIError


class ICloudCommands(CommandSet):
    """The Find My commands: ring a phone, locate its owner, list the account's devices."""

    @staticmethod
    def _get_icloud_api() -> iCloudAPI:
        return iCloudAPI.get_instance()

    def cmd_find_phone(self, parameter_string: str):
        """Answers at once and rings in the background: a session iCloud wants re-signed, 2FA
        code included, is waited out there (`play_sound`), so the reply is "pinging it" and the
        ring follows within the minute. The phone ringing is the confirmation; a ring that did
        not happen is told through a system note, which the next turn sees."""
        api = self._get_icloud_api()
        try:
            label = api.device_label(parameter_string)
        except iCloudAPIError as e:
            info(f"find_phone refused: {e}")
            return str(e)

        def _ring(name):
            try:
                api.play_sound(name)
            except Exception as e:
                error(f"find_phone: ringing {label} failed: {e}")
                system_note("iCloud", f"iCloud could not ring {label}: {e}")

        self.fire_and_forget(_ring, parameter_string)
        return (f"Ringing {label} now. If iCloud asks for a sign-in first it is completed in the background "
                f"and the phone rings within a minute; tell the user the ping is on its way, not that it is done.")

    def cmd_locate_user(self, parameter_string: str):
        self.background_command(self._get_icloud_api().get_location, parameter_string)

    def cmd_icloud_devices(self, parameter_string: str = ""):
        self.background_command(self._run_icloud_devices, "")

    def cmd_icloud_unlocked(self, parameter_string: str = ""):
        self.background_command(self._run_icloud_unlocked, "")

    def cmd_icloud_login_code(self, parameter_string: str):
        """The 2FA code the user read off their phone, relayed from the chat."""
        code = parameter_string.strip().strip("\"'`").strip()
        if not code:
            return 'icloud_login_code failed: give the 6-digit code Apple texted, e.g. icloud_login_code("123456")'
        self.background_command(self._run_icloud_login_code, code, lane=self.commands.CONFIG_LANE)

    def _run_icloud_login_code(self, code: str) -> str:
        return self._get_icloud_api().submit_user_code(code)

    def _run_icloud_unlocked(self, _="") -> str:
        """The user says the Apple Account works again: lift the park a refused sign-in put on
        the server's iCloud sign-ins and sign in now."""
        api = self._get_icloud_api()
        api.refresh_connection(by_user=True)
        status = api.get_auth_status()
        info(f"icloud_unlocked: signed in again — {status['status']}: {status['message']}")
        self.commands.integration_monitor().check_now(api.name)
        return f"iCloud sign-in attempted — {status['status']}: {status['message']}"

    def _run_icloud_devices(self, _="") -> str:
        """Every device the iCloud account sees, with its id and the icloud.devices entry it
        is configured in — the material for finishing the setup with config_set."""
        api = self._get_icloud_api()
        devices = api.get_icloud_devices(sign_in=True)
        if not devices:
            info("icloud_devices: the account listed no devices")
            return ("iCloud listed no devices — the login may be incomplete (list_integrations shows the "
                    "iCloud state; a 2FA code is entered in the console's iCloud tab).")
        lines = []
        for d in devices:
            model = f" ({d['model']})" if d.get("model") else ""
            if d.get("configured_as"):
                status = f"configured as '{d['configured_as']}' (icloud.devices[{d['slot']}])"
            else:
                status = "not configured"
            lines.append(f"{d['name']}{model} — id {d['id']} — {status}")
        n = api.configured_device_count()
        info(f"icloud_devices: {len(lines)} device(s), {n} configured")
        return "\n".join(lines) + (f"\n\nicloud.devices has {n} entr{'y' if n == 1 else 'ies'}; the next one is "
                                    f"icloud.devices[{n}] (keys id and display_name).")

    def describe_find_phone(self, p):             return "Pinging phone"
    def describe_locate_user(self, p):            return f"Locating {p.strip()}" if p.strip() else "Locating user"
    def describe_icloud_devices(self, p):         return "Listing iCloud devices"
    def describe_icloud_unlocked(self, p):        return "Signing in to iCloud again"
    def describe_icloud_login_code(self, p):      return "Entering the iCloud 2FA code"
