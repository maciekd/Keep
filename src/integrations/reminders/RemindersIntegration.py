import subprocess
import time
from dataclasses import dataclass
from typing import List, Tuple

from common.ConfigManager import config
from assistant.integration.Integration import Integration
from common.logging import error, info, trace


@dataclass(frozen=True)
class OpenReminder:
    """One reminder that is not completed yet, with the list it lives in. Lists are addressed
    by id: two lists may share a name (a local and a shared "Shoppinglista", say), and a
    name then always resolves to the first of them."""
    list_id: str
    list_name: str
    name: str


# Reminders answers any property read of a list in about 10-20 ms per reminder, completed
# ones included, so a list of a thousand takes a dozen seconds and a `whose` over the
# application as a whole never comes back on a full house. One osascript call per list keeps
# each within budget and names the list that failed.
_SCRIPT_TIMEOUT = 120

# One line per list, `<id>\t<name>`.
_LISTS_SCRIPT = '''
tell application "Reminders"
    set out to ""
    set allLists to every list
    repeat with i from 1 to count of allLists
        set l to item i of allLists
        set out to out & (id of l) & tab & (name of l) & linefeed
    end repeat
    return out
end tell
'''

# One line per open reminder of the list, its name. The names are fetched into a plain list
# before they are read: iterating a `whose` reference directly hands back references that
# osascript cannot coerce to text (error -1700).
_LIST_OPEN_SCRIPT = '''
tell application "Reminders"
    set openNames to name of (every reminder of list id "{list_id}" whose completed is false)
    set out to ""
    repeat with j from 1 to count of openNames
        set out to out & (item j of openNames) & linefeed
    end repeat
    return out
end tell
'''


def parse_lists(output: str) -> List[Tuple[str, str]]:
    """The lists script's output as (id, name) pairs; a line without the separator is skipped."""
    lists = []
    for line in output.splitlines():
        list_id, sep, name = line.partition("\t")
        if sep and list_id:
            lists.append((list_id, name))
        elif line:
            trace(f"Skipping unparsable lists line: {line!r}")
    return lists


def parse_open_reminders(output: str, list_id: str, list_name: str) -> List[OpenReminder]:
    """The listing script's output for one list as reminders, blank lines dropped. A title
    with a newline in it comes out as two reminders — rare enough to leave."""
    return [OpenReminder(list_id, list_name, line) for line in output.splitlines() if line.strip()]


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

    def _run_script(self, script: str, what: str) -> str:
        """The script's stdout; raises with the reason when osascript fails or hangs."""
        started = time.time()
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=_SCRIPT_TIMEOUT
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Reminders did not answer within {_SCRIPT_TIMEOUT}s reading {what}")
        trace(f"Reminders answered {what} in {time.time() - started:.1f}s")
        if result.returncode != 0:
            raise RuntimeError(f"{result.stderr.strip() or 'osascript failed'} (reading {what})")
        return result.stdout

    def lists(self) -> List[Tuple[str, str]]:
        """Every Reminders list as (id, name). Raises when Reminders cannot be asked."""
        lists = parse_lists(self._run_script(_LISTS_SCRIPT, "the lists"))
        trace(f"Reminders lists: {[name for _, name in lists]}")
        return lists

    def open_reminders(self) -> List[OpenReminder]:
        """Every reminder not completed yet, across all lists, one query per list. Raises
        when Reminders cannot be asked, naming the list that failed."""
        found = []
        for list_id, list_name in self.lists():
            script = _LIST_OPEN_SCRIPT.replace("{list_id}", _escape(list_id))
            in_list = parse_open_reminders(self._run_script(script, f'list "{list_name}"'), list_id, list_name)
            trace(f"Open reminders in '{list_name}': {len(in_list)}")
            found.extend(in_list)
        info(f"Open reminders: {len(found)} across all lists")
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
            f'tell application "Reminders" to tell list id "{_escape(target.list_id)}" '
            f'to set completed of (first reminder whose name is "{_escape(target.name)}" and completed is false) to true'
        )
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=_SCRIPT_TIMEOUT
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
