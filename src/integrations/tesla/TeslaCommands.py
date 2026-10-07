from assistant.commands.CommandSet import CommandSet
from integrations.tesla.TeslaIntegration import TeslaIntegration, TeslaAPIError


class TeslaCommands(CommandSet):
    """The cars' commands: max defrost, climate on and off, the trunk, a navigation destination."""
    MULTI_PARAM_COMMANDS = frozenset({"tesla_navigate"})

    # Looked up per call: get_instance() is the cache, so a restarted integration
    # (restart_integration) is what the next command uses.
    @staticmethod
    def _get_tesla_api() -> TeslaIntegration:
        return TeslaIntegration.get_instance()

    def _run_tesla_command(self, tesla_method, parameter_string: str) -> str:
        try:
            tesla_method(parameter_string if parameter_string else None)
            return ""
        except TeslaAPIError as e:
            return f"Tesla command failed: {e}"

    def cmd_tesla_max_defrost(self, parameter_string: str):
        self.background_command(
            lambda p: self._run_tesla_command(self._get_tesla_api().set_max_defrost, p),
            parameter_string
        )

    def cmd_tesla_air_con_on(self, parameter_string: str):
        self.background_command(
            lambda p: self._run_tesla_command(self._get_tesla_api().start_air_conditioning, p),
            parameter_string
        )

    def cmd_tesla_air_con_off(self, parameter_string: str):
        self.background_command(
            lambda p: self._run_tesla_command(self._get_tesla_api().stop_air_conditioning, p),
            parameter_string
        )

    def cmd_tesla_open_trunk(self, parameter_string: str):
        self.background_command(
            lambda p: self._run_tesla_command(self._get_tesla_api().open_trunk, p),
            parameter_string
        )

    @staticmethod
    def _navigation_args(args: tuple) -> tuple[str | None, str]:
        """(vehicle, destination) from the LLM's arguments: the vehicle leads and the rest is
        the destination, re-joined so an unquoted address keeps its commas; a lone argument
        is a destination for the default car."""
        parts = [a.strip() for a in args if a and a.strip()]
        if len(parts) == 1:
            return None, parts[0]
        return (parts[0] if parts else None), ", ".join(parts[1:])

    def cmd_tesla_navigate(self, *args: str):
        vehicle, destination = self._navigation_args(args)
        if not destination:
            return "tesla_navigate failed: expected 'vehicle_id, destination' parameters"
        self.background_command(self._run_tesla_navigation, (vehicle, destination))

    def _run_tesla_navigation(self, vehicle: str | None, destination: str) -> str:
        try:
            return self._get_tesla_api().navigate_to(vehicle, destination)
        except TeslaAPIError as e:
            return f"Tesla command failed: {e}"

    def describe_tesla_max_defrost(self, p):      return "Turning on Tesla max defrost"
    def describe_tesla_air_con_on(self, p):       return "Turning on Tesla A/C"
    def describe_tesla_air_con_off(self, p):      return "Turning off Tesla A/C"
    def describe_tesla_open_trunk(self, p):       return "Opening Tesla trunk"

    def describe_tesla_navigate(self, p):
        _vehicle, destination = self._navigation_args(tuple(p.split(",")) if p else ())
        return f"Sending Tesla navigation to {destination}" if destination else "Sending Tesla navigation destination"
