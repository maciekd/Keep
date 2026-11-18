from typing import List, Optional

from assistant.integration.Integration import HealthWarning, Integration
from assistant.agents.BuildPreflight import BuildPreflight, CREDITS_AGE_FOR_HEALTH, forget_credits
from assistant.agents.CodeQualityMaintainer import build_memory_warning
from common.logging import info, trace


class SelfImprovementIntegration(Integration):
    """Keep improving its own code with Claude Code, nightly and on request.

    Lets Keep analyse its own code, fix or build an integration as a pull request and report
    anything else to the developers as a GitHub issue, all through the Claude Code CLI on the
    server machine, with the GitHub CLI logged in for the PRs and the issues.

    The commands: self_analysis, improve_integration (a PR confined to one integration folder,
    or a new one), report_issue (an investigated GitHub issue) and, with the root config key
    is_developer true, self_improvement (a PR on any part of the code); the nightly
    code-quality runs are the integration's too. A gh account without write access opens its
    PRs from its own fork, and without gh an issue is drafted for the user to file by hand. The
    runs use api_keys.anthropic unless a claude.ai login is there.

    The switch and the health check: the runs (CodeQualityMaintainer) happen only while this
    integration is enabled, the nightly ones also only while `self_improvement.nightly_runs` is
    on. The setup is in the config.example.yaml beside this file, section self_improvement. The
    health check is the preflight a build runs before it starts (`BuildPreflight`), so
    everything a build would lack shows in list_integrations within a cycle and in the answer
    of enable_integration, all of it at once: without Claude Code, its credentials or usage
    for the jobs' model (asked with a one-word test request when the integration comes up,
    then once a day, and before a build) nothing runs and the check fails; with those in place and gh, its
    login or access to the repository missing the analyses still run, so the state is a
    warning that never becomes an error. A login takes precedence over the key
    (the key is only handed to the CLI when there is no login, since the environment variable
    would shadow a login and move its billing to the key). A machine with too little memory for
    a build is said when the integration is switched on (`enable_note`) and when a build starts,
    and is no health state."""

    def __init__(self):
        super().__init__()
        forget_credits()   # coming up is the setup's moment: the first health check asks for usage anew

    @property
    def name(self) -> str:
        return "Self Improvement"

    @property
    def config_keys(self) -> List[str]:
        return []

    def enable_note(self) -> Optional[str]:
        return build_memory_warning()

    def check_health(self) -> bool:
        preflight = BuildPreflight.check(credits_max_age=CREDITS_AGE_FOR_HEALTH)
        if not preflight.blocking:
            trace("Self Improvement: nothing a build needs is missing")
            return True
        if preflight.claude_down:
            raise RuntimeError(preflight.health_text())
        info("Self Improvement: Claude Code runs, a build would not start")
        raise HealthWarning("no build can start: " + preflight.health_text())
