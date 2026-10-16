"""The homeclaw_set and homeclaw_events commands: one verified write to any characteristic of
a HomeClaw accessory — the thermostat's target and mode, a cover's exact position, a light's
colour temperature — for what the house commands (turn_on / turn_off / dim / stop) do not
cover, and the app's event log as what changed in the house."""
import time

from assistant.commands.CommandSet import CommandSet
from assistant.integration.HomeAutomation import HomeAutomation
from common.logging import info
from integrations.homeclaw.HomeClawClient import HomeClawError
from integrations.homeclaw.HomeClawIntegration import HomeClawIntegration

DEFAULT_HOURS = 24
MAX_HOURS = 24 * 14


class HomeClawCommands(CommandSet):
    """Sets a characteristic of a HomeKit accessory through the HomeClaw app, and lists what changed."""

    MULTI_PARAM_COMMANDS = frozenset({"homeclaw_set"})
    SENSITIVE_COMMANDS = frozenset({"homeclaw_set"})
    PURE_COMMANDS = frozenset({"homeclaw_events"})
    APPROVAL_NOTES = {"homeclaw_set": "Writes a setting on a HomeKit accessory — a thermostat target, a mode, a position."}

    @staticmethod
    def _arg(value: str) -> str:
        return str(value or "").strip().strip("\"'`").strip()

    def cmd_homeclaw_set(self, *args: str):
        if len(args) < 3 or not all(self._arg(a) for a in args[:3]):
            return "homeclaw_set failed: expected 'name, characteristic, value' parameters"
        name = HomeAutomation.from_display(self._arg(args[0]))
        characteristic = self._arg(args[1]).lower().replace(" ", "_")
        value = self._arg(args[2]).removesuffix("°C").removesuffix("°").strip()
        self.background_command(lambda p: HomeClawIntegration.get_instance().set_characteristic(*p),
                                (name, characteristic, value))

    def cmd_homeclaw_events(self, parameter_string: str):
        text = self._arg(parameter_string).lower().removesuffix("h").removesuffix("hours").strip()
        try:
            hours = float(text) if text else DEFAULT_HOURS
        except ValueError:
            return f"homeclaw_events failed: '{parameter_string.strip()}' is not a number of hours"
        if hours <= 0:
            return "homeclaw_events failed: the hours must be above zero"
        self.background_command(self._run_homeclaw_events, min(hours, MAX_HOURS))

    @staticmethod
    def _run_homeclaw_events(hours: float) -> str:
        try:
            entries = HomeClawIntegration.get_instance().recent_events(hours)
        except (HomeClawError, TimeoutError) as e:
            info(f"homeclaw_events: {e}")
            return "Request timed out" if isinstance(e, TimeoutError) else str(e)
        if not entries:
            info(f"homeclaw_events: nothing changed in the last {hours:g} hours")
            return f"Nothing changed in the house in the last {hours:g} hours."
        lines = [f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(e.timestamp))}  {e.content}" for e in entries]
        return f"## HOUSE EVENTS, LAST {hours:g} HOURS\n" + "\n".join(lines)

    def describe_homeclaw_set(self, p):
        return f"Setting {p}" if p.strip() else "Setting a HomeKit characteristic"

    def describe_homeclaw_events(self, p):
        return "Reading what changed in the house"
