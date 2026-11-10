import re
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import List, Optional

from .SingletonMixin import SingletonMixin


class IntegrationState(Enum):
    """Health state of an external integration.

    Transitions: OK -> WARNING (1 failure) -> ERROR (3 consecutive failures).
    Any successful check resets back to OK. NOT_CONFIGURED is a terminal state
    for integrations whose config keys are missing.
    """
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    NOT_CONFIGURED = "not_configured"
    DISABLED = "disabled"           # not in config's enabled_integrations: never constructed
    NEEDS_2FA = "needs_2fa"
    RATE_LIMITED = "rate_limited"
    UNKNOWN = "unknown"


class HealthWarning(Exception):
    """Raised by `check_health` of an integration that works and still has something the owner
    should act on (the iMessage satellite could not update itself). The monitor shows the text
    with the state WARNING and does not count it towards ERROR, however often it repeats."""


class Integration(SingletonMixin, ABC):
    """Base class for the external service integrations (Tesla, Ring, YR, CalDAV, etc.).

    Subclasses implement name, config_keys, and check_health(). The IntegrationMonitor
    periodically calls check_health() and manages state transitions on this base class.
    Singleton access (get_instance/reset_instance) comes from SingletonMixin.

    A subclass's docstring is read by the console's Catalog tab, so it has three paragraphs:
    what the integration is in a few words (`description()`, the "Description" column), what
    it does for the user in one to three sentences (`details()`, shown with the selected
    integration), then the notes for whoever reads the code, which no UI shows.
    """

    # Off unless config.yaml lists the integration in `enabled_integrations`. The exceptions
    # are Web Access, which needs no credentials and the assistant leans on constantly, and
    # ElevenLabs TTS, the engine voice speaks through unless another one is enabled.
    enabled_by_default: bool = False
    # Its data is private (messages, mail, calendar, photos, cameras): the console's Catalog
    # tab tags it so, and the status of IntegrationMonitor carries the flag.
    sensitive: bool = False
    # What list_integrations says next to "needs 2FA": how the sign-in is done, when the
    # integration has a way of its own (a login command). Empty: the console's tab for it.
    # `current_login_hint` is the live version, for an integration whose way depends on the
    # moment (iCloud: whether something on this Mac reads the texted code).
    login_hint: str = ""
    # The integration to sign in before this one, by its config name, and why: a disabled
    # integration's line in `list_integrations` says so after its enable command, and
    # `enable_integration` warns when the prerequisite is not healthy while setup is in
    # progress. The first-run brief orders the steps too, but only the opening turn carries it
    # whole — the first betas enabled Photos next to iMessage, before iCloud was signed in.
    setup_after: str = ""
    setup_after_reason: str = ""

    def current_login_hint(self) -> str:
        """The login hint for the state the integration is in now; the class attribute unless
        the integration knows better at the moment."""
        return self.login_hint

    @classmethod
    def wipe_targets(cls) -> list[tuple[Path, str]]:
        """The files of a signed-in state the integration keeps in the tree, each with its
        description — what `//wipe` backs up and deletes beside the core's own targets."""
        return []

    @classmethod
    def log_out(cls) -> Optional[str]:
        """Called by `//wipe` before its `wipe_targets` go, so a session that stays authorized
        on the account after its file is deleted is ended there first. Returns what happened,
        for the wipe's listing; None when there is nothing to end."""
        return None

    @classmethod
    def config_name(cls) -> str:
        """The name an integration is listed under in `enabled_integrations`, readable before
        an instance exists: the constant `name` property, else the class name without
        'Integration' (ArloCameraIntegration -> ArloCamera; give such a class a constant name)."""
        prop = getattr(cls, "name", None)
        if isinstance(prop, property):
            try:
                value = prop.fget(None)
                if isinstance(value, str) and value:
                    return value
            except Exception:
                pass
        return cls.__name__.removesuffix("Integration")

    @classmethod
    def _paragraphs(cls) -> List[str]:
        """The paragraphs of the class docstring, each on one line (a class without a docstring has none)."""
        return [" ".join(part.split()) for part in re.split(r"\n\s*\n", cls.__doc__ or "") if part.strip()]

    @classmethod
    def description(cls) -> str:
        """What the integration is, in a few words, readable before an instance exists: the
        first paragraph of the class docstring ("" for a class without one)."""
        return next(iter(cls._paragraphs()), "")

    @classmethod
    def details(cls) -> str:
        """What the integration does for the user, in one to three sentences: the second
        paragraph of the class docstring ("" for a docstring of one paragraph)."""
        return next(iter(cls._paragraphs()[1:]), "")

    @classmethod
    def enabled(cls, config) -> bool:
        return config.integration_enabled(cls.config_name(), cls.enabled_by_default)

    def __init__(self):
        self._state: IntegrationState = IntegrationState.UNKNOWN
        self._last_check_time: Optional[float] = None
        self._last_error: Optional[str] = None
        self._consecutive_failures: int = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this integration, e.g. "Tesla" or "Ring"."""
        ...

    @property
    @abstractmethod
    def config_keys(self) -> List[str]:
        """Config paths required for this integration (checked during registration).
        If any key is missing, the integration is marked NOT_CONFIGURED."""
        ...

    @abstractmethod
    def check_health(self) -> bool:
        """Return True if the integration is reachable and functional.
        May raise an exception on failure — the monitor will catch it. `HealthWarning` is the
        one for an integration that works and has something to report all the same."""
        ...

    def wait_ready(self, timeout: float) -> bool:
        """Block until a freshly started instance can answer `check_health` truthfully, or
        `timeout` runs out; True when it can. A restart checks health the moment the start
        hook returns, and an integration that comes up on a thread of its own (Web Access's
        Chrome) would otherwise report a warning that stands until the monitor's next cycle,
        15 minutes later. Ready on construction by default."""
        return True

    def missing_config_keys(self) -> List[str]:
        """The config_keys that config.yaml does not set (absent or empty) — what the
        monitor's NOT_CONFIGURED verdict rests on, for a caller that wants the list."""
        from common.ConfigManager import config
        missing = []
        for key in self.config_keys:
            try:
                value = config.get(key)
            except KeyError:
                value = None
            if value in (None, "", [], {}):   # an empty accounts list or section is not set either
                missing.append(key)
        return missing

    @property
    def state(self) -> IntegrationState:
        """Current health state, managed by IntegrationMonitor."""
        return self._state

    @state.setter
    def state(self, value: IntegrationState):
        self._state = value

    @property
    def last_check_time(self) -> Optional[float]:
        """Epoch timestamp of the most recent health check, or None if never checked."""
        return self._last_check_time

    @property
    def last_error(self) -> Optional[str]:
        """Error message from the most recent failure, or None if last check passed."""
        return self._last_error

    def start(self) -> None:
        """Begins the integration's own background work, a thread of its own: the monitor runs
        it right after registration and again on every restart, in place included, so what
        runs is replaced, and `_on_reset` is where it stops. Nothing by default. A Feed's
        thread is FeedRunner's, started by the feed hook instead of this."""
        pass

    def needs_2fa(self) -> bool:
        """Return True if this integration requires 2FA before it can operate.
        Must be lightweight — only inspect local state, never trigger 2FA emails."""
        return False

    def on_not_configured(self):
        """Called when the integration is marked NOT_CONFIGURED during registration.
        Override to clean up persisted state (tokens, sessions, etc.)."""
        pass

    def on_needs_2fa(self):
        """Called when the integration requires 2FA. Override to attempt auto-resolution
        (e.g. polling email for verification codes)."""
        pass

    def accounts_status(self) -> list[dict]:
        """One dict per account an account-holding integration (mail, calendar) has — name,
        state, last_error — for list_integrations and the Catalog tab; none for the rest."""
        return []

    @property
    def consecutive_failures(self) -> int:
        """Number of consecutive failed health checks. Resets to 0 on success."""
        return self._consecutive_failures
