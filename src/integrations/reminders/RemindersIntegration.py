import subprocess
from typing import List, Optional

from common.ConfigManager import config
from assistant.integration.Integration import Integration
from common.logging import error
from integrations.reminders.RemindersStore import RemindersStore

DEFAULT_LIST_NAME = "Michelle"


class RemindersIntegration(Integration):
    """Apple Reminders: the shopping list, and Keep's own alarmed reminders.

    Adds items to the family's shopping list when asked, and keeps reminders of its own in a
    Reminders list that syncs to the user's iPhone: one with a due time rings there at that
    time, one with a place rings on arriving at or leaving it, and Keep sees which were done.

    The shopping list goes through AppleScript (`add_reminder_item`, unchanged); the own list
    (`reminders.list_name`, created on first use) through EventKit in `RemindersStore`, since
    AppleScript cannot set a location alarm. The health check is the AppleScript side; the
    EventKit side reports its own permission state through the command replies."""
    sensitive = True

    @property
    def name(self) -> str:
        return "Reminders"

    @property
    def config_keys(self) -> List[str]:
        return ["reminders"]

    def __init__(self):
        super().__init__()
        self._store: Optional[RemindersStore] = None

    @property
    def list_name(self) -> str:
        return config.get("reminders.list_name") or DEFAULT_LIST_NAME

    @property
    def store(self) -> RemindersStore:
        """The EventKit side, on Keep's own list; rebuilt when the configured name changes."""
        if self._store is None or self._store.list_name != self.list_name:
            self._store = RemindersStore(self.list_name)
        return self._store

    def check_health(self) -> bool:
        result = subprocess.run(
            ["osascript", "-e", 'tell application "Reminders" to get name of default list'],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0

    def add_reminder_item(self, item: str, list_name: str = None) -> str:
        if not list_name:
            list_name = config.get("reminders.shopping_list_name")

        escaped = item.replace("\\", "\\\\").replace('"', '\\"')
        escaped_list = list_name.replace("\\", "\\\\").replace('"', '\\"')
        script = (
            f'tell application "Reminders" to tell list "{escaped_list}" '
            f'to make new reminder with properties {{name:"{escaped}"}}'
        )

        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                error(f"osascript failed: {result.stderr.strip()}")
                return f"Failed to add reminder: {result.stderr.strip()}"
            return ""
        except Exception as e:
            error(f"Failed to add reminder: {e}")
            return f"Failed to add reminder: {e}"


def add_reminder_item(item: str, list_name: str = None) -> str:
    return RemindersIntegration.get_instance().add_reminder_item(item, list_name)
