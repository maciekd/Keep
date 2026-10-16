from datetime import datetime

from assistant.commands.CommandSet import CommandSet
from integrations.homeclaw.HomeClawClient import HomeClawError
from integrations.homeclaw.HomeClawIntegration import HomeClawIntegration
from integrations.homeclaw.HomeClawModel import Unresolved

DRY_RUN = {"dry_run", "dry-run", "dryrun", "dry_run=true", "dry run"}
ON, OFF = {"on", "true", "1", "open", "start"}, {"off", "false", "0", "close", "closed", "stop"}


class HomeClawCommands(CommandSet):
    """The HomeKit accessories behind HomeClaw: covers, the garage door, valves, the thermostat, lights, scenes, the event log, the automations."""

    MULTI_PARAM_COMMANDS = frozenset({"homekit_cover", "homekit_garage", "homekit_valve", "homekit_thermostat",
                                      "homekit_light", "homekit_scene"})
    SENSITIVE_COMMANDS = frozenset({"homekit_garage", "homekit_thermostat"})
    APPROVAL_NOTES = {
        "homekit_garage": "Opens or closes the garage door.",
        "homekit_thermostat": "Changes the temperature the thermostat heats to.",
    }

    @staticmethod
    def _split(args) -> tuple[list[str], bool]:
        """The arguments without the dry-run flag, and whether it was there."""
        values = [str(a).strip() for a in args if str(a).strip()]
        dry = [a for a in values if a.lower() in DRY_RUN]
        return [a for a in values if a.lower() not in DRY_RUN], bool(dry)

    @staticmethod
    def _switch(word: str, what: str) -> bool:
        word = word.lower()
        if word in ON:
            return True
        if word in OFF:
            return False
        raise Unresolved(f"Not a {what}: {word!r}")

    @staticmethod
    def _position(word: str) -> int:
        word = word.lower().removesuffix("%").strip()
        if word in ("open", "up"):
            return 100
        if word in ("close", "closed", "down"):
            return 0
        try:
            position = round(float(word.replace(",", ".")))
        except ValueError:
            raise Unresolved(f"Not a position: {word!r} — use open, close or 0–100")
        if not 0 <= position <= 100:
            raise Unresolved(f"A position is 0–100 (% open), not {position}")
        return position

    @staticmethod
    def _celsius(word: str) -> float:
        try:
            return float(word.replace(",", ".").removesuffix("°C").removesuffix("C").strip())
        except ValueError:
            raise Unresolved(f"Not a temperature: {word!r}")

    def _run(self, action, args):
        def work(*params):
            values, dry_run = self._split(params)
            try:
                return action(HomeClawIntegration.get_instance(), values, dry_run)
            except (HomeClawError, Unresolved) as e:
                return str(e)
        self.background_command(work, args)

    @staticmethod
    def _pair(values: list[str], usage: str) -> tuple[str, str]:
        """(ref, value) of a two-argument command whose ref may be left out."""
        if len(values) == 1:
            return "", values[0]
        if len(values) == 2:
            return values[0], values[1]
        raise Unresolved(f"Expected {usage}")

    def cmd_homekit_cover(self, *args):
        def act(hc, values, dry):
            ref, position = self._pair(values, "homekit_cover(ref, open|close|0-100)")
            return hc.set_cover(ref, self._position(position), dry)
        self._run(act, args)

    def cmd_homekit_garage(self, *args):
        def act(hc, values, dry):
            ref, word = self._pair(values, "homekit_garage(ref, open|close)")
            return hc.set_garage(ref, self._switch(word, "door command (open or close)"), dry)
        self._run(act, args)

    def cmd_homekit_valve(self, *args):
        def act(hc, values, dry):
            ref, word = self._pair(values, "homekit_valve(ref, on|off)")
            return hc.set_valve(ref, self._switch(word, "valve command (on or off)"), dry)
        self._run(act, args)

    def cmd_homekit_thermostat(self, *args):
        def act(hc, values, dry):
            ref, word = self._pair(values, "homekit_thermostat(ref, celsius)")
            return hc.set_thermostat(ref, self._celsius(word), dry)
        self._run(act, args)

    def cmd_homekit_light(self, *args):
        def act(hc, values, dry):
            ref, word = self._pair(values, "homekit_light(ref, on|off)")
            return hc.set_light(ref, self._switch(word, "light command (on or off)"), dry)
        self._run(act, args)

    def cmd_homekit_scene(self, *args):
        def act(hc, values, dry):
            if not values:
                raise Unresolved("Expected homekit_scene(name)")
            return hc.trigger_scene(", ".join(values), dry)
        self._run(act, args)

    def cmd_homekit_events(self, parameter_string: str):
        self.background_command(self._events, self.commands.first_argument(parameter_string or "") or "1h")

    @staticmethod
    def _events(since: str) -> str:
        try:
            entries = HomeClawIntegration.get_instance().events(since)
        except HomeClawError as e:
            return str(e)
        if not entries:
            return f"No HomeKit changes since {since} (sensor readings and light tweaks are not logged)."
        return "\n".join(f"{datetime.fromtimestamp(e.timestamp):%Y-%m-%d %H:%M} {e.content.removeprefix('HomeKit: ')}"
                         for e in entries)

    def cmd_homekit_automations(self, parameter_string: str):
        self.background_command(self._automations, self.commands.first_argument(parameter_string or ""))

    @staticmethod
    def _automations(wanted: str) -> str:
        try:
            return HomeClawIntegration.get_instance().describe_automations(wanted)
        except HomeClawError as e:
            return str(e)

    def describe_homekit_cover(self, p):       return f"Moving the HomeKit cover ({p})"
    def describe_homekit_garage(self, p):      return f"Garage door via HomeClaw ({p})"
    def describe_homekit_valve(self, p):       return f"Irrigation valve via HomeClaw ({p})"
    def describe_homekit_thermostat(self, p):  return f"Setting the HomeKit thermostat ({p})"
    def describe_homekit_light(self, p):       return f"Switching the HomeKit light ({p})"
    def describe_homekit_scene(self, p):       return f"Running the HomeKit scene {p}"
    def describe_homekit_events(self, p):      return "Reading the HomeKit event log"
    def describe_homekit_automations(self, p): return "Reading the HomeKit automations"
