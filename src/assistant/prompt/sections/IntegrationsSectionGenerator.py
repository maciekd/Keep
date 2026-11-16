"""The INTEGRATIONS section: what is on, on but unconfigured, and on but broken, in the
prompt while setup is in progress (`FirstRun.in_ftue`) and absent otherwise. Without it the
assistant knows its own state only from a `list_integrations` result in the history window,
which a trim takes away: the first beta install's assistant then told the user to create the
Telegram bot it had set up itself an hour earlier. Names only — which keys are set is
`list_integrations`' business — so it costs a few dozen tokens. Polled, so an integration the
user enables, or breaks, reaches the LLM as a delta note. The last line is the Full Disk Access
grant (`common/FullDiskAccess`), the one setup step made at the Mac itself: the assistant decides
from it whether the step is ahead, done, or failed after the restart. A server that is not a Mac
has no such grant: its last line names the system instead, and what needs a Mac."""
from ..PromptSectionGenerator import PromptSectionGenerator
from common import FullDiskAccess, Platform
from common.logging import info


class IntegrationsSectionGenerator(PromptSectionGenerator):
    """Renders the enabled integrations grouped by state while setup is in progress, nothing after."""

    UPDATE_INTERVAL = 20

    HEADING = "# INTEGRATIONS"
    # The health states (IntegrationState values) under each line; "disabled" is left out.
    ON = ("ok", "unknown")
    NOT_CONFIGURED = ("not_configured",)
    PROBLEM = {"warning": "warning", "error": "error", "needs_2fa": "needs 2FA", "rate_limited": "rate limited"}
    ERROR_CHARS = 80

    def __init__(self, prompt_manager):
        super().__init__(prompt_manager, "Integrations")
        self._in_ftue = None

    @staticmethod
    def in_ftue() -> bool:
        from assistant import FirstRun
        return FirstRun.in_ftue(verbose=False)

    @staticmethod
    def statuses() -> list[dict]:
        from assistant.integration.IntegrationMonitor import IntegrationMonitor
        return IntegrationMonitor.get_instance().get_all_statuses()

    @staticmethod
    def full_disk_access() -> bool | None:
        """Granted, not granted, or None when the probe cannot tell (no Messages database)."""
        if not FullDiskAccess.observable():
            return None
        return FullDiskAccess.granted()

    def _fetch_content(self) -> str:
        """Polled every UPDATE_INTERVAL, so the verdict and the summary are logged only when
        they change — the previous content is the thing to compare with."""
        in_ftue = self.in_ftue()
        if in_ftue != self._in_ftue:
            self._in_ftue = in_ftue
            info("Integrations section: setup is in progress — the section is in the prompt" if in_ftue
                 else "Integrations section: setup is not in progress — nothing in the prompt")
        if not in_ftue:
            return ""
        statuses = self.statuses()
        access = self.full_disk_access()
        text = self.render(statuses, access)
        if text != self.content:
            on, unconfigured, broken = self.group(statuses)
            info(f"Integrations section: {len(on)} on, {len(unconfigured)} not configured, {len(broken)} with a problem; "
                 + (f"Full Disk Access {self.ACCESS[access]}" if Platform.is_mac() else f"on {Platform.label()}"))
        return text

    @classmethod
    def group(cls, statuses: list[dict]) -> tuple[list[str], list[str], list[str]]:
        """The names by state: on, on but unconfigured, on with a problem (with its detail)."""
        on, unconfigured, broken = [], [], []
        for status in sorted(statuses, key=lambda s: s["name"].lower()):
            state = status["state"]
            if state in cls.ON:
                on.append(status["name"])
            elif state in cls.NOT_CONFIGURED:
                unconfigured.append(status["name"])
            elif state in cls.PROBLEM:
                detail = cls.PROBLEM[state]
                if status.get("last_error"):
                    detail += f": {status['last_error'][:cls.ERROR_CHARS]}"
                broken.append(f"{status['name']} ({detail})")
        return on, unconfigured, broken

    ACCESS = {True: "granted", False: "not granted", None: "not checkable (no Messages database on this Mac)"}

    @classmethod
    def render(cls, statuses: list[dict], full_disk_access: bool | None = None) -> str:
        on, unconfigured, broken = cls.group(statuses)
        lines = [cls.HEADING]
        lines.append("On: " + (", ".join(on) + "." if on else "nothing yet."))
        if unconfigured:
            lines.append("On but not configured: " + ", ".join(unconfigured) + ".")
        if broken:
            lines.append("On with a problem: " + ", ".join(broken) + ".")
        if Platform.is_mac():
            lines.append(f"Full Disk Access for this server process: {cls.ACCESS[full_disk_access]}.")
        else:
            lines.append(cls.needs_a_mac(statuses))
        return "\n".join(lines)

    LOCAL_AUDIO = "the server's own microphones and speakers"

    @classmethod
    def needs_a_mac(cls, statuses: list[dict]) -> str:
        """The platform line of a server that is not a Mac: what cannot be set up on it."""
        names = sorted((s["name"] for s in statuses if s.get("needs_mac")), key=str.lower)
        return (f"The server runs on {Platform.label()}; {', '.join([*names, cls.LOCAL_AUDIO])} need a Mac — "
                f"never offer them.")
