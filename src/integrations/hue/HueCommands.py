"""The hue_pair command: pairs Keep with the Hue bridge from any chat. The bridge hands out an
application key only while its link button has just been pressed, so the command asks for the
button when it was not, and on success writes the key (and the bridge address it found) to
config.yaml and restarts the integration — nothing for the user to type."""
import requests

from assistant.commands.CommandSet import CommandSet
from common import ConfigStore
from common.ConfigManager import config
from common.logging import info
from .HueClient import HueError, LinkButtonNotPressed, pair
from .HueDiscovery import discover

INTEGRATION = "Philips Hue"
PRESS_BUTTON = ("press the round link button on the bridge, then run hue_pair() again within 30 seconds")


class HueCommands(CommandSet):
    """Pairs with the Philips Hue bridge from the chat."""

    def cmd_hue_pair(self, parameter_string: str):
        host = self.commands.first_argument(parameter_string)
        self.background_command(self._run_hue_pair, (host,), lane=self.commands.CONFIG_LANE)

    def _run_hue_pair(self, host: str) -> str:
        monitor = self.commands.integration_monitor()
        integration = monitor.find(INTEGRATION)
        if integration is None:
            info("hue_pair: Philips Hue is not registered")
            return f"{INTEGRATION} is not available — restart the server."
        label = self.commands.state_label(integration.state.value, "not available")
        if label == "disabled":
            info("hue_pair: Philips Hue is disabled")
            return f'{INTEGRATION} is disabled — enable_integration("{INTEGRATION}") first, then hue_pair().'
        host = (host or str(config.get("hue.bridge", "") or "")).strip()
        if not host:
            found = discover()
            if not found:
                info("hue_pair: neither mDNS nor the discovery service found a bridge")
                return ('hue_pair failed: no bridge found on this network. Ask the user for its address (the Hue app '
                        'shows it under Settings → My Hue system → the bridge) and run hue_pair("<address>").')
            if len(found) > 1:
                return f"hue_pair: several bridges found ({', '.join(found)}) — run hue_pair(\"<address>\") with the right one."
            host = found[0]
        try:
            key = pair(host)
        except LinkButtonNotPressed:
            return f"The link button on the bridge at {host} has not been pressed: ask the user to {PRESS_BUTTON}."
        except (HueError, requests.RequestException) as e:
            info(f"hue_pair: pairing with {host} failed: {e}")
            return f"hue_pair failed: {e}"
        ConfigStore.set_value("hue.app_key", key)
        if str(config.get("hue.bridge", "") or "").strip() != host:
            ConfigStore.set_value("hue.bridge", host)
        result = monitor.restart(INTEGRATION)
        state = self.commands.describe_restart(result["status"], result["restartable"]) if result else ""
        return f"{INTEGRATION} is paired with the bridge at {host}. {state}".strip()

    def describe_hue_pair(self, p):
        host = self.commands.first_argument(p)
        return f"Pairing with the Hue bridge at {host}" if host else "Pairing with the Hue bridge"
