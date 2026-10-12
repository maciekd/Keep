import subprocess
from dataclasses import dataclass
from typing import List

from common.ConfigManager import config
from assistant.integration.Integration import Integration
from common.logging import error, info, trace


@dataclass(frozen=True)
class OpenReminder:
    """One reminder that is not completed yet, with the list it lives in."""
    list_name: str
    name: str


# One line per open reminder, `<list>\t<name>`; a reminder named with a tab or a newline is
# rare enough that the parser simply skips a line without the separator.
_OPEN_REMINDERS_SCRIPT = '''
tell application "Reminders"
    set out to ""
    repeat with l in lists
        set ln to name of l
        repeat with rn in (name of every reminder of l whose completed is false)
            set out to out & ln & tab & rn & linefeed
        end repeat
    end repeat
    return out
end tell
'''


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


class RemindersIntegration(Integration):
    """Apple Reminders lists: the shopping list added to, reminders ticked off.

    Adds items to Apple Reminders lists, such as the shopping list, and marks a reminder as
    completed when asked.

    Goes through AppleScript."""
    sensitive = True

    @property
    def name(self) -> str:
        return "Reminders"

    @property
    def config_keys(self) -> List[str]:
        return ["reminders"]

    def check_health(self) -> bool:
        result = subprocess.run(
            ["osascript", "-e", 'tell application "Reminders" to get name of default list'],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0

    def add_reminder_item(self, item: str, list_name: str = None) -> str:
        if not list_name:
            list_name = config.get("reminders.shopping_list_name")

        script = (
            f'tell application "Reminders" to tell list "{_escape(list_name)}" '
            f'to make new reminder with properties {{name:"{_escape(item)}"}}'
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

    def open_reminders(self) -> List[OpenReminder]:
        """Every reminder not completed yet, across all lists. Raises when Reminders cannot
        be asked."""
        result = subprocess.run(
            ["osascript", "-e", _OPEN_REMINDERS_SCRIPT],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "osascript failed")
        found = []
        for line in result.stdout.splitlines():
            list_name, sep, name = line.partition("\t")
            if sep and name:
                found.append(OpenReminder(list_name, name))
        trace(f"Open reminders: {len(found)} across all lists")
        return found

    def complete_reminder(self, text: str) -> str:
        """Marks the one open reminder whose title is `text` (case-insensitive) as completed.
        Returns "" when done, the reason otherwise — no match, or several, which it names."""
        wanted = text.strip()
        if not wanted:
            return "complete_reminder failed: no reminder text given"
        try:
            candidates = [r for r in self.open_reminders() if r.name.strip().casefold() == wanted.casefold()]
        except Exception as e:
            error(f"Could not list open reminders: {e}")
            return f"Failed to read the reminders: {e}"

        if not candidates:
            info(f"No open reminder matches '{wanted}'")
            return f"No open reminder is called \"{wanted}\""
        if len(candidates) > 1:
            named = ", ".join(f'"{r.name}" in list "{r.list_name}"' for r in candidates)
            info(f"{len(candidates)} open reminders match '{wanted}', refusing: {named}")
            return (f"{len(candidates)} open reminders match \"{wanted}\" — {named}. "
                    "Not completing any; ask which one is meant.")

        target = candidates[0]
        script = (
            f'tell application "Reminders" to tell list "{_escape(target.list_name)}" '
            f'to set completed of (first reminder whose name is "{_escape(target.name)}" and completed is false) to true'
        )
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                error(f"osascript failed: {result.stderr.strip()}")
                return f"Failed to complete reminder: {result.stderr.strip()}"
            info(f"Completed reminder '{target.name}' in list '{target.list_name}'")
            return ""
        except Exception as e:
            error(f"Failed to complete reminder: {e}")
            return f"Failed to complete reminder: {e}"


def add_reminder_item(item: str, list_name: str = None) -> str:
    return RemindersIntegration.get_instance().add_reminder_item(item, list_name)


def complete_reminder(text: str) -> str:
    return RemindersIntegration.get_instance().complete_reminder(text)
