from assistant.commands.CommandSet import CommandSet
from integrations.reminders.RemindersIntegration import add_reminder_item, complete_reminder


class RemindersCommands(CommandSet):
    """Adding to the shared shopping list in Reminders, and ticking a reminder off by its title."""

    def cmd_add_shopping_list_item(self, parameter_string: str) -> str:
        return add_reminder_item(parameter_string)

    def describe_add_shopping_list_item(self, p): return f"Adding {p.strip()} to the shopping list"

    def cmd_complete_reminder(self, parameter_string: str) -> str:
        return complete_reminder(parameter_string)

    def describe_complete_reminder(self, p): return f"Completing the reminder {p.strip()}"
