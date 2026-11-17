from assistant.commands.CommandSet import CommandSet
from integrations.ferry.FerryIntegration import FerryError, FerryIntegration


class FerryCommands(CommandSet):
    """The ferry's next departures, both directions or from one harbor."""

    PURE_COMMANDS = frozenset({"ferry_departures"})

    def cmd_ferry_departures(self, parameter_string: str = ""):
        def read(origin):
            try:
                return FerryIntegration.get_instance().departures(origin)
            except FerryError as e:
                return str(e)
        self.background_command(read, self.commands.first_argument(parameter_string or ""))

    def describe_ferry_departures(self, p):
        origin = self.commands.first_argument(p or "")
        return f"Checking the ferry from {origin}" if origin else "Checking the ferry"
