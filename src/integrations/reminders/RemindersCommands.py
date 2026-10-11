from assistant.commands.CommandSet import CommandSet
from common.logging import error
from integrations.reminders.RemindersIntegration import RemindersIntegration, add_reminder_item
from integrations.reminders.RemindersStore import RemindersAccessError, format_listing, DUE_FORMAT, DATE_FORMAT
from datetime import datetime


class RemindersCommands(CommandSet):
    """Adding to the shared shopping list, and Keep's own reminders: created with a time or a
    geofence alarm, listed with what was completed, completed and deleted."""
    MULTI_PARAM_COMMANDS = frozenset({"create_reminder"})

    def cmd_add_shopping_list_item(self, parameter_string: str) -> str:
        return add_reminder_item(parameter_string)

    def describe_add_shopping_list_item(self, p): return f"Adding {p.strip()} to the shopping list"

    def cmd_create_reminder(self, *args: str):
        self.background_command(self._run_create_reminder, list(args))

    def cmd_list_reminders(self, _: str = ""):
        self.background_command(self._run_list_reminders, [])

    def cmd_complete_reminder(self, reminder_id: str):
        self.background_command(self._run_complete_reminder, [self.commands.first_argument(reminder_id)])

    def cmd_delete_reminder(self, reminder_id: str):
        self.background_command(self._run_delete_reminder, [self.commands.first_argument(reminder_id)])

    @staticmethod
    def _store():
        return RemindersIntegration.get_instance().store

    @classmethod
    def _run_create_reminder(cls, *args: str) -> str:
        args = [a.strip() for a in args]
        if not args or not args[0]:
            return "create_reminder failed: expected (title, due, notes, place) — only the title is required"
        title, due, notes, place = (args + ["", "", "", ""])[:4]
        try:
            made = cls._store().create(title, due, notes, place)
        except ValueError as e:
            return f"create_reminder failed: {e}"
        except RemindersAccessError as e:
            return f"create_reminder failed: {e}"
        except Exception as e:
            error(f"create_reminder failed: {type(e).__name__}: {e}")
            return f"create_reminder failed: {type(e).__name__}: {e}"
        store = cls._store()
        when = ""
        if made.due:
            when = " due " + made.due.strftime(DUE_FORMAT if made.timed else DATE_FORMAT)
            when += " (alarm on the phone at that time)" if made.timed else " (a day, no alarm)"
        where = f", alarm at {made.place}" if made.place else ""
        return f"Created reminder {made.id} '{made.title}' in the '{store.list_name}' list{when}{where}."

    @classmethod
    def _run_list_reminders(cls) -> str:
        try:
            store = cls._store()
            now = datetime.now()
            return format_listing(store.list(now), store.list_name, now)
        except RemindersAccessError as e:
            return f"list_reminders failed: {e}"
        except Exception as e:
            error(f"list_reminders failed: {type(e).__name__}: {e}")
            return f"list_reminders failed: {type(e).__name__}: {e}"

    @classmethod
    def _run_complete_reminder(cls, reminder_id: str) -> str:
        try:
            done = cls._store().complete(reminder_id)
            return f"Completed reminder {done.id} '{done.title}'."
        except RemindersAccessError as e:
            return f"complete_reminder failed: {e}"
        except Exception as e:
            error(f"complete_reminder failed: {type(e).__name__}: {e}")
            return f"complete_reminder failed: {type(e).__name__}: {e}"

    @classmethod
    def _run_delete_reminder(cls, reminder_id: str) -> str:
        try:
            gone = cls._store().delete(reminder_id)
            return f"Deleted reminder {gone.id} '{gone.title}'."
        except RemindersAccessError as e:
            return f"delete_reminder failed: {e}"
        except Exception as e:
            error(f"delete_reminder failed: {type(e).__name__}: {e}")
            return f"delete_reminder failed: {type(e).__name__}: {e}"

    def describe_create_reminder(self, p):
        title = self.commands.first_argument(p)
        return f"Creating reminder '{title}'" if title else "Creating a reminder"

    def describe_list_reminders(self, p): return "Listing the reminders"
    def describe_complete_reminder(self, p): return f"Completing reminder {self.commands.first_argument(p)}"
    def describe_delete_reminder(self, p): return f"Deleting reminder {self.commands.first_argument(p)}"
