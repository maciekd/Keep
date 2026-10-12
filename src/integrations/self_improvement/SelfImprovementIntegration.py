from typing import List, Optional

from assistant.integration.Integration import Integration
from assistant.agents.CodeQualityMaintainer import claude_status, ClaudeStatus, API_KEY_CONFIG_KEY
from assistant.SystemNote import system_note
from common.logging import info, trace

NOTE_SOURCE = "Self Improvement"
API_KEY_WARNING = (
    f"Self Improvement: Claude Code on the server is not logged in to claude.ai, so every run "
    f"(self_analysis, improve_integration, report_issue and the nightly runs) is billed per token on "
    f"{API_KEY_CONFIG_KEY} — far more expensive than a subscription. To get back on the subscription, "
    f"open a terminal on the server machine itself (not over SSH), run `claude` and complete /login; "
    f"the next health check (within 15 minutes) picks the login up."
)


class SelfImprovementIntegration(Integration):
    """Keep improving its own code with Claude Code, nightly and on request.

    Lets Keep analyse its own code, fix or build an integration as a pull request and report
    anything else to the developers as a GitHub issue, all through the Claude Code CLI on the
    server machine, with the GitHub CLI logged in for the PRs and the issues.

    The commands: self_analysis, improve_integration (a PR confined to one integration folder,
    or a new one), report_issue (an investigated GitHub issue) and, with the root config key
    is_developer true, self_improvement (a PR on any part of the code); the nightly
    code-quality runs are the integration's too. The runs use api_keys.anthropic unless a
    claude.ai login is there.

    The switch and the health check: the runs (CodeQualityMaintainer) happen only while this
    integration is enabled, the nightly ones also only while `self_improvement.nightly_runs` is
    on. The setup is in the config.example.yaml beside this file, section self_improvement. The
    health check asks `claude auth status`, a free local call, so a missing CLI or one without
    credentials shows in list_integrations within a cycle. A login takes precedence over the key
    (the key is only handed to the CLI when there is no login, since the environment variable
    would shadow a login and move its billing to the key). The key is billed per token, so the
    health check tells the household (a system note, in the history and the chats) when it
    finds the key standing in for a login — once, when the credentials change, not every
    cycle — and again when a login is back after a spell on the key."""

    def __init__(self):
        super().__init__()
        self._on_api_key: Optional[bool] = None   # what the last check saw; None before the first

    @property
    def name(self) -> str:
        return "Self Improvement"

    @property
    def config_keys(self) -> List[str]:
        return []

    def check_health(self) -> bool:
        status = claude_status()
        if not status.ok:
            raise RuntimeError(status.detail)
        trace(f"Self Improvement: Claude Code {status.detail}")
        self._note_credentials(status)
        return True

    def _note_credentials(self, status: ClaudeStatus):
        """Warn the household when the runs are on the API key — the first check that sees
        it, then only when the credentials change — and say so when a login is back."""
        if status.api_key == self._on_api_key:
            trace("Self Improvement: Claude Code credentials unchanged, nothing to tell")
            return
        was_on_key = self._on_api_key
        self._on_api_key = status.api_key
        if status.api_key:
            info(f"Self Improvement: Claude Code has no claude.ai login, the runs are billed on "
                 f"{API_KEY_CONFIG_KEY} — warning the household")
            system_note(NOTE_SOURCE, API_KEY_WARNING)
        elif was_on_key:
            info(f"Self Improvement: Claude Code is {status.detail} again, off the API key — telling the household")
            system_note(NOTE_SOURCE, f"Self Improvement: Claude Code is {status.detail} again, so its runs are "
                                     f"back on the subscription and no longer billed on {API_KEY_CONFIG_KEY}.")
        else:
            trace(f"Self Improvement: Claude Code {status.detail} from the start, nothing to tell")
