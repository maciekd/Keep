import hashlib
import json
import logging
import pathlib
import shutil
import threading
import time
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from assistant.SystemNote import system_note

from pyicloud import PyiCloudService
from pyicloud.base import resolve_cookie_directory
from pyicloud.exceptions import (PyiCloudFailedLoginException, PyiCloudAPIResponseException,
                                 PyiCloudAuthRequiredException, PyiCloudConnectionException,
                                 PyiCloudAcceptTermsException, PyiCloud2FARequiredException)
from pyicloud.services.findmyiphone import FindMyiPhoneServiceManager

from common.ConfigManager import ConfigManager
from common.Paths import ICLOUD_SESSION_DIR
from assistant.tools.Geocoding import reverse_geocode
from common.RateLimiter import RateLimiter
from assistant.integration.Integration import Integration, IntegrationState
from assistant.integration import SmsCodeReader
from common.logging import info, error, note, trace, fine

_EXC_LOG_LIMIT = 300


# The system row a code fished out of iMessage and accepted by Apple leaves, and the system
# turn that opens the conversation with it during the first-time setup (iCloudAPI._announce_verified).
VERIFIED_NOTE = "iCloud 2FA code auto-detected from iMessage and verified. ✅"
VERIFIED_TURN = (VERIFIED_NOTE + " iCloud is signed in. Tell the user in a line and go on with the next step "
                 "of the setup.")


def _brief(e: BaseException, limit: int = _EXC_LOG_LIMIT) -> str:
    """An exception rendered for one log line. pyicloud appends the entire HTTP body
    to its API errors — on 2026-09-06 that put Apple's 940-line 2FA page into
    floxy.out — so collapse the whitespace and cut the tail, keeping the reason
    and status code that lead the message. It also hides Apple's answer behind a
    generic message of its own — on 2026-10-15 "Invalid email/password combination."
    stood for a 403 saying the account was locked — so the chained causes follow."""
    parts: List[str] = []
    exc: Optional[BaseException] = e
    while exc is not None and len(parts) < 3:
        text = " ".join(str(exc).split())
        if text and not any(text in part for part in parts):
            parts.append(text)
        exc = exc.__cause__
    text = " <- ".join(parts)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… [{len(text)} chars, {type(e).__name__}]"


def _request_sms_code_only(api: PyiCloudService) -> None:
    """Replacement for PyiCloudService.request_2fa_code (2026.10; `_request_2fa_code`
    up to 2.6.x): request the SMS code for the pending challenge without the
    trusted-device push."""
    from pyicloud.const import CONTENT_TYPE_JSON
    phone = api._trusted_phone_number()
    if phone is None:
        trace("iCloud: signin challenge has no trusted phone — not requesting an SMS code")
        return
    try:
        api.session.put(
            f"{api._auth_endpoint}/verify/phone",
            json={"phoneNumber": phone.as_phone_number_payload(), "mode": "sms"},
            headers=api._get_auth_headers({"Accept": CONTENT_TYPE_JSON}),
        )
        trace(f"iCloud: requested SMS 2FA code for phone id {phone.device_id} (no device push)")
    except Exception as e:
        info(f"iCloud: SMS code request after signin failed ({_brief(e)}) — Apple usually pushes one anyway")


def _refresh_find_my_without_reauth(manager: FindMyiPhoneServiceManager, locate: bool, retry: bool = False) -> None:
    """Replacement for FindMyiPhoneServiceManager._refresh_client_with_reauth (2026.10): the
    original answers Find My's "Re-authentication required" with a password sign-in on the
    spot, inside whatever device read hit it, and with no one watching for the 2FA challenge
    it opens — the Oct 6–13 sign-ins that ended on a challenge nothing completed. Here the
    refusal propagates and ICloudIntegration decides whether to sign in
    (`_run_with_reauthentication`: one deliberate sign-in, its challenge completed, none while
    sign-ins are parked)."""
    manager.stop_event.set()
    manager._refresh_client(locate=locate)
    manager._initialize_devices(locate=locate)
    if manager._refresh_interval is not None:
        manager._start_monitor_thread()


FindMyiPhoneServiceManager._refresh_client_with_reauth = _refresh_find_my_without_reauth


_original_authenticate_with_token = PyiCloudService._authenticate_with_token


def _authenticate_with_token_logged(api: PyiCloudService, require_trust: bool = True) -> None:
    """Wrapper around PyiCloudService._authenticate_with_token (2026.10): say what Apple
    answered the token login with. It is the first step of every sign-in; when it does not
    yield a trusted session pyicloud goes on to a password sign-in, and whether Apple then
    raises a 2FA challenge depends on the same trust. On 2026-10-16 two such sign-ins ended
    on a challenge with a trust token from the same morning in the session file, and the log
    could not tell a refused session token from a browser Apple no longer trusts."""
    trust_token_sent = bool(api.session.data.get("trust_token"))
    try:
        _original_authenticate_with_token(api, require_trust)
    except PyiCloud2FARequiredException:
        info("iCloud: token login answered, but Apple does not trust the browser "
             f"(hsaTrustedBrowser={api.data.get('hsaTrustedBrowser')!r}, "
             f"trust token sent: {trust_token_sent}) — password sign-in next")
        raise
    except PyiCloudFailedLoginException as e:
        cause = e.__cause__
        response = getattr(cause, "response", None)
        status = getattr(cause, "code", None) or getattr(response, "status_code", None)
        info(f"iCloud: token login refused (HTTP {status}, {_brief(cause) if cause else _brief(e)}; "
             f"trust token sent: {trust_token_sent}) — password sign-in next")
        raise
    trace(f"iCloud: token login ok (hsaTrustedBrowser={api.data.get('hsaTrustedBrowser')!r}, "
          f"trust token sent: {trust_token_sent})")


PyiCloudService._authenticate_with_token = _authenticate_with_token_logged


_MFA_PAGE_STATUSES = (200, 409)


def _get_mfa_auth_options_tolerant(api: PyiCloudService) -> Dict[str, Any]:
    """Replacement for PyiCloudService._get_mfa_auth_options (2.6.x): read the 2FA
    page Apple serves after a password signin whether it arrives with 200 or 409.
    Apple answers 409 for 'signin incomplete, second factor pending' on its other
    auth endpoints too, and on 2026-09-06 it served this page that way — complete,
    boot args included. pyicloud's normal request path turns any 409 into
    'Authentication required for Account.' and aborts the signin before the SMS
    step, so fetch the page raw and parse whatever came back."""
    from pyicloud.hsa2_bridge import Hsa2BootContext, parse_boot_args_html
    headers = api._get_auth_headers({"Accept": "text/html"})
    response = api.session.request_raw("GET", api._auth_endpoint, headers=headers)
    status = response.status_code
    if status not in _MFA_PAGE_STATUSES:
        raise PyiCloudAPIResponseException(
            response.reason or "2FA page request failed", status, response)

    try:
        response_json = response.json()
    except (AttributeError, TypeError, ValueError):
        response_json = None

    auth_options: Dict[str, Any] = {}
    if isinstance(response_json, dict):
        auth_options.update(response_json)
        boot_context = Hsa2BootContext.from_auth_options(auth_options)
    else:
        boot_context = parse_boot_args_html(response.text or "")
    if status != 200:
        info(f"iCloud: Apple served the 2FA page with HTTP {status} — parsed it anyway "
             f"(route={boot_context.auth_initial_route!r}, factors={list(boot_context.auth_factors)})")

    auth_options.update(boot_context.as_auth_data())
    api._hsa2_boot_context = boot_context
    api._clear_trusted_device_bridge_state()
    api._set_two_factor_delivery_state("unknown")
    return auth_options


class _PyiCloudLoggingHandler(logging.Handler):
    """Routes pyicloud's stdlib logging into Floxy's logging system.
    pyicloud logs the interesting 2FA details (request URLs, 'Code verification
    failed.', 'Session trust failed.') to its own logger, which otherwise never
    reaches floxy.out."""

    def emit(self, record):
        if record.getMessage().startswith("Saved "):
            return  # "Saved session/cookies data to file" after every request: a save that failed warns on its own
        msg = self.format(record)
        if record.levelno >= logging.ERROR:
            error(f"[pyicloud] {msg}")
        elif record.levelno >= logging.INFO:
            trace(f"[pyicloud] {msg}")
        else:
            fine(f"[pyicloud] {msg}")


def _setup_pyicloud_logging():
    handler = _PyiCloudLoggingHandler()
    handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))
    logger = logging.getLogger('pyicloud')
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


_setup_pyicloud_logging()


class iCloudAPIError(Exception):
    """Base class for iCloud API related errors."""


class iCloudConfigurationError(iCloudAPIError):
    """Raised when required iCloud configuration is missing or invalid."""


class iCloudAuthenticationError(iCloudAPIError):
    """Raised when authentication fails or 2FA is required."""


@dataclass
class iCloudDeviceSpec:
    id: str  # Find My's `id` (what pyicloud keys devices on); older entries hold the baUUID
    display_name: str  # Human-friendly name for voice commands

    def label(self) -> str:
        """Return the best label to use for human facing messages."""
        return self.display_name

    def matches(self, data: Dict[str, Any]) -> bool:
        """Whether a Find My record is this device. The record's `id` is the identity; the
        baUUID is accepted for entries written before 2026-10, when a phone re-enrolled in
        Find My showed the field can go blank while the record itself stays."""
        return self.id in (data.get("id"), data.get("baUUID"))


@dataclass
class FindMyProbe:
    """What one read-only Find My request said: `ok` (Find My answers on the cookies the
    session holds; `devices` are the account's records), `reauth` (Apple wants a fresh
    sign-in before Find My serves this session) or `error` (anything else; `detail` says
    what). `complete` marks a listing that also polled for the family members' devices."""
    verdict: str
    detail: str = ""
    devices: List[Dict[str, Any]] = field(default_factory=list)
    complete: bool = False
    at: float = field(default_factory=time.time)


# The body of the Find My web page's own device-list request (pyicloud sends the same),
# minus the `shouldLocate` / `isUpdatingAllLocations` flags that make every device in the
# account take a fresh GPS fix.
_FIND_MY_CLIENT_CONTEXT = {
    "appName": "iCloud Find (Web)", "appVersion": "2.0", "apiVersion": "3.0",
    "deviceListVersion": 1, "fmly": True, "timezone": "US/Pacific", "inactiveTime": 0,
}


class ICloudIntegration(Integration):
    """Find My: where the family's devices are, and a ring for a lost phone.

    Locates the family's devices through iCloud Find My and makes a lost phone ring. Signing in
    takes the Apple ID and the 2FA code Apple texts.

    Owns the pyicloud session, including automatic SMS 2FA recovery with strict caps on code
    requests and fresh signins so a broken flow can never spam Apple or the phone. Apple's risk
    engine locked the account on 2026-10-15 and -16 after days of half-finished sign-ins and
    a Find My read every 15 minutes that mostly failed, so the rules since are: nothing talks to
    Find My on a timer (a command does), every password sign-in is one this class runs and
    completes with the code Apple texts (pyicloud's own re-auth is disabled), and a refusal from
    Apple parks sign-ins — in a file, across restarts — until the user confirms the account is
    usable again. A park is tied to the credentials it was earned with: a new apple_id or
    password lifts it, since a typo on the first set-up is not a security refusal.

    The auto-2FA flow needs a reader of the Mac's text messages (`SmsCodeReader`, iMessage):
    with none live no code is requested on a timer — the user enters the one Apple sent, through
    the assistant (icloud_login_code) or the console's iCloud tab."""
    sensitive = True
    login_hint = ("Apple texted a 6-digit code and nothing on this Mac reads text messages (the iMessage feed "
                  "is not running), so the user enters it: they tell it to the assistant, which runs "
                  "icloud_login_code(code), or type it in the console's iCloud tab (Ctrl-E)")
    AUTO_CODE_LOGIN_HINT = ("Apple texted a 6-digit code to the trusted phone; the iMessage feed reads it from "
                            "Messages on this Mac and the server submits it by itself — nothing for the user to "
                            "do unless a minute passes; then they tell the assistant the code (icloud_login_code)")

    SESSION_EXPIRED_2FA_MESSAGE = (
        "iCloud is waiting for a 2FA code: the server submits the one Apple texts on its own, so try again "
        "in a minute (manual entry: the console's iCloud tab)."
    )
    NO_DEVICES_MESSAGE = (
        "No iCloud devices are configured yet: icloud_devices() lists the devices the account sees "
        "and the config keys that add one."
    )

    # pyicloud's AppleDevice.location is a cached property — it never asks the
    # device for a fresh fix. Without an active locate we only ever see Apple's
    # last-known position, which can be many minutes stale (isOld). When a fix
    # comes back stale we trigger a locate and poll for up to this long; the
    # device responds asynchronously so it needs a few seconds to report in.
    _LOCATE_TIMEOUT_SECONDS = 10.0
    _LOCATE_POLL_INTERVAL_SECONDS = 2.0

    # Automatic SMS 2FA: cap unattended code requests so a broken flow can't
    # SMS-bomb the phone; give a requested code this long to arrive before
    # considering it lost and allowing a new request.
    AUTO_SMS_MAX_REQUESTS = 4
    AUTO_SMS_WINDOW_SECONDS = 30 * 60
    SMS_CODE_WAIT_SECONDS = 5 * 60
    # After a failed recovery (e.g. Apple throttling SMS with -28248), back off
    # for this long. The limiter alone can't stop the spam: the monitor retries
    # every 15 min, which stays under 4-per-30-min forever — each retry doing a
    # full SRP signin and firing a Telegram alert.
    AUTO_SMS_FAILURE_COOLDOWN_SECONDS = 2 * 60 * 60
    # How long a ring or a location lookup waits for the auto-2FA flow to answer the challenge
    # its sign-in opened: the code arrived and was accepted 21 s after the challenge on
    # 2026-10-24; before this the command failed at once and the phone never rang.
    CHALLENGE_WAIT_SECONDS = 60
    # Hard cap on from-scratch password (SRP) signins per recovery, independent
    # of the SMS limiter: on 2026-08-24 a 409 on every submitted code drove four
    # password logins + four SMS in 63s before the limiter kicked in.
    MAX_FRESH_SIGNINS_PER_RECOVERY = 1
    # The read-only Find My probe behind the TUI's device table and the device listing: one
    # verdict serves them for this long. Nothing runs it on a timer: until 2026-10-16 the health
    # check did, every 15 minutes, and three of four answered "Re-authentication required"
    # (Find My's cookies live 15-30 minutes) — some 75 failed authentications a day that Apple
    # read as an unknown client holding the password.
    FIND_MY_PROBE_TTL_SECONDS = 5 * 60
    FAMILY_POLLS = 3            # refreshClient rounds a complete listing waits for family devices
    FAMILY_POLL_DELAY_SECONDS = 0.5
    # Blocks only the user can clear, by the text Apple's answer carries, and what to tell them.
    APPLE_BLOCKS = (
        ("terms of service",
         "iCloud is blocked until you accept Apple's updated terms of service. "
         "Sign in at https://icloud.com (or open Settings on an Apple device) and accept them."),
        ("locked for security reasons",
         "Apple has locked the Apple Account for security reasons, so the server's iCloud sign-ins fail "
         "(Find My on the phones keeps working). Unlock it at https://iforgot.apple.com."),
    )
    # A refusal from Apple (a locked account, a rejected password, new terms) parks every
    # automatic sign-in until the user says the account works again — a retry against a locked
    # account is what the risk engine reads as an attack, and the production machine restarts
    # 17-23 times a day, each start a sign-in. The verdict lives in this file beside the session
    # so the restarts keep it; Ctrl-R in the iCloud tab and icloud_unlocked delete it.
    PARK_FILE = "parked.json"
    PARK_NOTICE = ("The server will not sign in to iCloud again until you confirm the account works: "
                   "tell the assistant (it runs icloud_unlocked) or press Ctrl-R in the console's iCloud tab. "
                   "A corrected apple_id or password lifts it by itself.")
    CODE_WAITING_NOTICE = ("iCloud signed in and Apple texted a 6-digit code to the account's trusted phone. "
                           "The server reads it from Messages on this Mac and submits it by itself — nothing to do "
                           "unless a minute passes; then the user tells the assistant the code (icloud_login_code).")
    CODE_NEEDED_NOTICE = ("iCloud signed in and Apple texted a 6-digit code to the account's trusted phone. "
                          "Nothing on this Mac reads text messages (the iMessage feed is not running), so the user "
                          "enters it: tell the assistant the code (it runs icloud_login_code) or type it in the "
                          "console's iCloud tab (Ctrl-E).")

    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._config_manager = config_manager or ConfigManager()
        self._config = self._load_config(config)
        self._validate_config()
        self._find_my_probe: Optional[FindMyProbe] = None
        self._probe_lock = threading.Lock()
        self._api: Optional[PyiCloudService] = None
        self._auth_error: Optional[str] = None
        self._park_memory: Optional[str] = None     # the park when its file could not be written
        self._code_noted_for: Optional[float] = None  # the sign-in whose challenge the chats were told about
        # One Find My sign-in at a time: the LLM runs its commands in parallel, and two that
        # found Find My's cookies rejected at once each ran a password sign-in on the one
        # session until 2026-10-17. The ones that waited take the outcome of the one that ran.
        self._find_my_signin_lock = threading.Lock()
        self._find_my_signins = 0
        self._find_my_signin_outcome: Optional[str] = None
        self._auto_2fa_message: Optional[str] = None
        self._sms_requested_at: Optional[float] = None
        self._auto_sms_limiter = RateLimiter(self.AUTO_SMS_MAX_REQUESTS, self.AUTO_SMS_WINDOW_SECONDS)
        self._auto_sms_last_failure_ts = 0.0
        # Serializes auto SMS requests: the photo downloader's worker threads can
        # all hit an auth error at once, and without the lock each would fire its
        # own SMS request before the first one registers as pending.
        self._auto_sms_lock = threading.Lock()
        self._stale_session_rejection = False
        # True while a 2FA challenge one of our own password signins opened is still
        # unanswered. Only such a challenge accepts a code — codes requested on
        # token-restored sessions get rejected with 409 'Authentication required for
        # Account' (observed 2026-08-20), so requesting on them wastes an SMS — and it
        # is the whole of `requires_2fa()`: a session pyicloud merely calls untrusted
        # has nothing a code could answer.
        # Set when a code is accepted, cleared when a challenge opens — what `_wait_for_find_my` waits on.
        self._challenge_answered = threading.Event()
        self._challenge_from_fresh_signin = False
        self._fresh_signins_this_recovery = 0
        # True only while Floxy runs a signin it intends to complete with a code
        # (startup auth, fresh recovery signin, the Find My re-auth). pyicloud calls
        # request_2fa_code by itself after ANY SRP signin that ends in a 2FA
        # challenge — including the Find My re-auth it ran inside every device
        # listing until 2026-10-15 (now _sign_in_for_find_my, deliberate and
        # spaced), which sent one SMS per server restart (Aug 31–Sep 1) that
        # nothing ever submitted.
        self._signin_sms_wanted = False

    @property
    def _api(self) -> Optional[PyiCloudService]:
        return self.__api

    @_api.setter
    def _api(self, api: Optional[PyiCloudService]):
        """A new or dropped pyicloud session makes the last Find My verdict moot."""
        self.__api = api
        self._find_my_probe = None

    @property
    def _challenge_from_fresh_signin(self) -> bool:
        return self.__challenge_from_fresh_signin

    @_challenge_from_fresh_signin.setter
    def _challenge_from_fresh_signin(self, open_: bool):
        """A challenge opening restarts the wait of the commands that need it answered."""
        self.__challenge_from_fresh_signin = open_
        if open_:
            self._challenge_answered.clear()

    @property
    def name(self) -> str:
        return "iCloud"

    @property
    def config_keys(self) -> List[str]:
        return ["icloud"]

    def check_health(self) -> bool:
        """Signed in, with no 2FA challenge open. Nothing here touches Find My: until 2026-10-16
        the check read the device list every 15 minutes and re-signed in once an hour when the
        cookies had expired, and the stream of refusals is the likeliest road to the two
        lockouts. Find My is read when a command needs it, and a session it wants re-signed is
        idle, not unhealthy."""
        self._ensure_authenticated()
        return not self.requires_2fa()

    def needs_2fa(self) -> bool:
        return self.requires_2fa()

    def _load_config(self, override_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if override_config is not None:
            return override_config

        try:
            return self._config_manager.get("icloud")
        except KeyError as exc:
            raise iCloudConfigurationError(
                "Missing 'icloud' section in configuration."
            ) from exc

    def _validate_config(self):
        if not self._config.get("apple_id"):
            raise iCloudConfigurationError(
                "Missing iCloud configuration value: apple_id"
            )
        if not self._config.get("password"):
            raise iCloudConfigurationError(
                "Missing iCloud configuration value: password"
            )

        if not self._config.get("devices"):
            info("iCloud: no devices configured — the session still starts (photos, device listing); "
                 "find_phone / locate_user need icloud.devices (icloud_devices() shows how to fill it)")

    @property
    def _device_specs(self) -> List[iCloudDeviceSpec]:
        """The usable configured devices; empty when none are configured yet. A half-written
        entry (id set, display_name still to come) is skipped, not fatal."""
        entries = self._config.get("devices") or []
        if not entries:
            return []
        specs: List[iCloudDeviceSpec] = []
        for entry in entries:
            if not isinstance(entry, dict):
                error(f"Invalid iCloud device configuration entry: {entry}")
                continue
            if not entry.get("id"):
                error(f"iCloud device missing 'id': {entry}")
                continue
            if not entry.get("display_name"):
                error(f"iCloud device missing 'display_name': {entry}")
                continue
            specs.append(
                iCloudDeviceSpec(
                    id=entry.get("id"),
                    display_name=entry.get("display_name"),
                )
            )
        if not specs:
            error("No valid iCloud device configuration entries were found.")
        return specs

    def device_label(self, device_name: Optional[str]) -> str:
        """The configured name of the device *device_name* picks — the iCloudConfigurationError
        of an unknown name, raised here, is what a command answers at once."""
        return self._select_device_spec(device_name).label()

    def _select_device_spec(self, device_name: Optional[str]) -> iCloudDeviceSpec:
        """Find device spec by display_name (used in voice commands)."""
        specs = self._device_specs
        if not specs:
            raise iCloudConfigurationError(self.NO_DEVICES_MESSAGE)
        if device_name:
            target = device_name.strip().lower()
            for spec in specs:
                if target in spec.display_name.lower() or spec.display_name.lower() in target:
                    return spec
            raise iCloudConfigurationError(
                f"No iCloud device matches '{device_name}'. Available: {', '.join(s.display_name for s in specs)}"
            )
        return specs[0]

    @staticmethod
    def _session_dir() -> str:
        """Where pyicloud persists the session token, trust token and cookies.
        pyicloud defaults to the per-user temp dir, which macOS wipes on reboot;
        every reboot then cost a from-scratch password signin plus a 2FA round
        (2026-08-17/20/30). One-time migration: adopt the temp-dir session if the
        persistent dir is still empty, so the switch itself does not force a signin."""
        target = ICLOUD_SESSION_DIR
        target.mkdir(parents=True, exist_ok=True)
        if not any(target.iterdir()):
            legacy = pathlib.Path(resolve_cookie_directory())
            files = [f for f in legacy.iterdir() if f.is_file()] if legacy.is_dir() else []
            for f in files:
                shutil.copy2(f, target / f.name)
            if files:
                info(f"iCloud: migrated {len(files)} session file(s) from {legacy} to {target}")
            else:
                trace(f"iCloud: no session files to migrate from {legacy}")
        return str(target)

    def _signin_2fa_code_hook(self, api: PyiCloudService) -> None:
        """Replacement for pyicloud's automatic post-signin code request: SMS-only
        (pyicloud's own starts with the trusted-device push, which pops an approval
        dialog on every trusted Apple device), and only when Floxy is running a
        signin it intends to complete with a code — pyicloud fires this after ANY
        SRP signin ending in a 2FA challenge, service re-auths included."""
        if not self._signin_sms_wanted:
            info("iCloud: suppressed pyicloud's automatic post-signin SMS request — "
                 "no 2FA recovery in progress (likely a service re-auth on a "
                 "working-but-untrusted session)")
            return
        _request_sms_code_only(api)

    def _new_service(self, authenticate: bool = True) -> PyiCloudService:
        # refresh_interval=None: no monitor thread re-reading Find My every five minutes for
        # the rest of the process once a command has used it (a command rebuilds the manager).
        api = PyiCloudService(
            self._config["apple_id"], self._config["password"],
            cookie_directory=self._session_dir(), authenticate=False, refresh_interval=None)
        api.request_2fa_code = types.MethodType(self._signin_2fa_code_hook, api)
        api._get_mfa_auth_options = types.MethodType(_get_mfa_auth_options_tolerant, api)
        if authenticate:
            api.authenticate()
        return api

    def _auto_sent_phone(self) -> Optional[Dict[str, Any]]:
        """The trusted phone Apple pushed a code to as part of the current
        password signin, or None when the session was restored from a token
        (no live challenge) or the challenge is not an SMS one."""
        api = self._api
        auth_data = getattr(api, "_auth_data", None) if api else None
        if not isinstance(auth_data, dict):
            return None
        pv = auth_data.get("phoneNumberVerification")
        phone = pv.get("trustedPhoneNumber") if isinstance(pv, dict) else None
        if isinstance(phone, dict) and phone.get("pushMode") == "sms":
            return phone
        return None

    def _arm_auto_sent_code(self, signin_started_at: float) -> bool:
        """After a password signin that ended in a 2FA challenge, treat the
        code Apple delivers on its own (pushMode=sms; pyicloud 2.6.x also
        requests it explicitly right after SRP) as the pending SMS request, so
        the iMessage poller submits it. Until 2026-08-30 the flow PUT a second
        /verify/phone request on top of that, and Apple answered the code
        with 409 every time — plus the extra text. Returns True when armed."""
        api = self._api
        if not api or not api.requires_2fa:
            return False
        phone = self._auto_sent_phone()
        if not phone:
            trace("iCloud: signin left no SMS challenge to wait on")
            return False
        api._auth_data["trustedPhoneNumber"] = phone
        api._two_factor_delivery_method = "sms"
        self._sms_requested_at = signin_started_at
        info(f"iCloud: password signin needs 2FA — waiting for the code Apple sends to "
             f"{phone.get('obfuscatedNumber', '???')} on its own (no extra SMS request)")
        self._note_challenge(signin_started_at)
        return True

    def current_login_hint(self) -> str:
        """The 2FA hint the setup commands answer with: with a code reader live the user has
        nothing to type — the FTUE assistant told the user to type the code the reader was about
        to submit (2026-10-19), because the class hint cannot know about the reader."""
        return self.AUTO_CODE_LOGIN_HINT if self._reader_live() else self.login_hint

    def _reader_live(self) -> bool:
        """Whether something on this Mac reads the codes Apple texts (the iMessage feed). Quiet:
        the status rows the console redraws every frame ask it through `current_login_hint`, so
        the verdict is logged where it decides something, not here (2026-10-20)."""
        return bool(SmsCodeReader.live())

    def _note_challenge(self, signin_started_at: float):
        """One system row per challenge: the chats (and the LLM) learn that a code is on its way
        and whether the server reads it by itself or the user has to enter it. While setup is in
        progress the row is not pushed to the chats: the challenge comes from the sign-in a setup
        command of the assistant is running, and its reply, seconds later, says it in the user's
        terms — the raw note on Telegram read as the LLM's instructions (2026-10-19)."""
        if self._code_noted_for == signin_started_at:
            return
        self._code_noted_for = signin_started_at
        from assistant import FirstRun
        in_ftue = FirstRun.in_ftue(verbose=False)
        if in_ftue:
            info("iCloud: setup is in progress — the challenge note is a history row only, the assistant's reply carries it")
        reader_live = self._reader_live()
        info("iCloud: the 2FA code is read by the Mac's messages" if reader_live
             else "iCloud: nothing reads the Mac's text messages now — the 2FA code must be entered by the user")
        system_note(self.name, self.CODE_WAITING_NOTICE if reader_live else self.CODE_NEEDED_NOTICE,
                    push=not in_ftue)

    def _ensure_authenticated(self):
        """Ensure we have an authenticated API connection: the session files when they still
        serve, a password sign-in otherwise — unless sign-ins are parked, when nothing is tried."""
        if self._api is not None:
            return
        parked = self._parked_reason()
        if parked:
            self._auth_error = f"sign-ins parked after Apple refused one: {parked}"
            info(f"iCloud: not signing in — {self._auth_error}")
            raise iCloudAuthenticationError(self._auth_error)

        try:
            info("Authenticating to iCloud using user id "+self._config["apple_id"])
            started_at = time.time()
            self._signin_sms_wanted = True
            try:
                self._api = self._new_service()
            finally:
                self._signin_sms_wanted = False
            self._auth_error = None
            self._fresh_signins_this_recovery = 0
            # A token-restored session carries no live challenge; only a password
            # signin (no/expired session file) does — and that one is as good as
            # any fresh signin, its auto-sent code included.
            self._challenge_from_fresh_signin = self._challenge_open()
            self._arm_auto_sent_code(started_at)
            info("iCloud authentication successful"
                 + (" — a 2FA challenge is open, the auto-2FA flow completes it" if self._challenge_from_fresh_signin else ""))
        except (PyiCloudFailedLoginException, PyiCloudAcceptTermsException) as e:
            self._auth_error = _brief(e)
            error(f"iCloud: Failed to authenticate: {_brief(e)}")
            self._park(e)
            raise iCloudAuthenticationError(f"Failed to authenticate: {e}") from e
        except PyiCloudConnectionException as e:
            self._auth_error = _brief(e)
            error(f"iCloud: Failed to authenticate: {_brief(e)}")
            raise iCloudAuthenticationError(f"Failed to authenticate: {e}") from e

    def _challenge_open_now(self) -> bool:
        """Whether a code could be answered right now: a challenge one of our sign-ins opened,
        not pyicloud's flag alone — a token-restored session Apple merely calls untrusted carries
        it too, and on that one a "code texted" notice is false and every code is "rejected"
        (2026-10-19, the install with no SMS reader). No sign-in is run to find out."""
        return self._api is not None and self._api.requires_2fa and self._challenge_from_fresh_signin

    def _challenge_open(self) -> bool:
        """Whether the sign-in that just ran ended on a 2FA challenge — pyicloud holds the
        challenge page in `_auth_data`; a token-restored session it merely calls untrusted
        has none, and `requires_2fa` alone does not tell the two apart."""
        api = self._api
        if api is None or not api.requires_2fa:
            return False
        auth_data = getattr(api, "_auth_data", None)
        return isinstance(auth_data, dict) and bool(auth_data)

    def _park_path(self) -> pathlib.Path:
        return pathlib.Path(self._session_dir()) / self.PARK_FILE

    def _credentials_fingerprint(self) -> str:
        """A digest of the apple_id and password, so the park file can tell whether the user
        has changed them since — never the values themselves."""
        raw = f"{self._config.get('apple_id', '')}\n{self._config.get('password', '')}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _parked_reason(self) -> Optional[str]:
        """Why sign-ins are parked — Apple's refusal and when — or None. A park earned with
        other credentials than the current ones is lifted here: the user corrected them, and a
        retry with a new password is not what the risk engine objects to."""
        if self._park_memory:
            return self._park_memory
        path = self._park_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            if data.get("credentials") and data["credentials"] != self._credentials_fingerprint():
                path.unlink(missing_ok=True)
                info("iCloud: apple_id or password changed since Apple refused the sign-in — park lifted")
                return None
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(data["at"])))
            return f"{data['reason']} ({when})"
        except (ValueError, KeyError, TypeError, OSError):
            return "a refused sign-in (the details were lost)"

    @staticmethod
    def _http_status(exc: BaseException) -> Optional[int]:
        """The HTTP status the exception or one of its causes carries, or None."""
        e: Optional[BaseException] = exc
        while e is not None:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if not isinstance(status, int):
                status = getattr(e, "code", None)
            if isinstance(status, int) and 100 <= status < 600:
                return status
            e = e.__cause__
        return None

    def _park(self, exc: BaseException):
        """Apple refused a sign-in: park every automatic one until the user confirms the account
        works again (`refresh_connection(by_user=True)` lifts it), and tell the user once where
        to go — a block Apple names (new terms, a security lock; 2026-10-15: HTTP 403, code
        -20209, which pyicloud 2.6.5 reported as a wrong password) gets its own message. A 5xx
        is Apple's side failing, not a refusal: pyicloud wraps it in the same login exception,
        and parking on it (until 2026-10-17) left the server signed out until the user noticed."""
        reason = _brief(exc)
        status = self._http_status(exc)
        if status is not None and status >= 500:
            info(f"iCloud: Apple answered the sign-in with HTTP {status} — a fault on Apple's side, "
                 f"not a refusal, so sign-ins are not parked ({reason})")
            return
        if self._parked_reason():
            trace(f"iCloud: sign-ins already parked, not repeating the notice ({reason})")
            return
        try:
            self._park_path().write_text(json.dumps({"at": time.time(), "reason": reason,
                                                     "credentials": self._credentials_fingerprint()}))
        except OSError as e:
            self._park_memory = reason
            error(f"iCloud: could not write the park file ({e}) — parked in memory, until the next restart")
        lowered = _brief(exc, limit=10_000).lower()
        message = next((text for marker, text in self.APPLE_BLOCKS if marker in lowered),
                       f"Apple refused the server's iCloud sign-in ({reason}).")
        info(f"iCloud: Apple refused the sign-in — automatic sign-ins parked until the user confirms ({reason})")
        system_note(self.name, f"{message} {self.PARK_NOTICE}")

    def _lift_park(self):
        parked = self._parked_reason()
        if not parked:
            trace("iCloud: no park to lift")
            return
        self._park_memory = None
        self._park_path().unlink(missing_ok=True)
        info(f"iCloud: the user confirmed the account works — park lifted (was: {parked})")

    def _probe_find_my(self, force: bool = False, complete: bool = False) -> FindMyProbe:
        """One read of the Find My device list on the cookies the session holds — the request
        the Find My web page makes when it opens, without the locate flags (no device is woken)
        and outside pyicloud's device manager. The verdict is cached FIND_MY_PROBE_TTL_SECONDS
        so the TUI's table and the device listing share one request; `complete` also polls for
        the family members' devices, which Apple hands out after the account's own. Nothing
        calls this on a timer."""
        with self._probe_lock:
            cached = self._find_my_probe
            # Only an ok verdict carries a device list that can be incomplete; a reauth or an
            # error inside the TTL is served as it is — repeating the refused request is what
            # got the account locked
            if (cached and not force and (cached.verdict != "ok" or cached.complete or not complete)
                    and time.time() - cached.at < self.FIND_MY_PROBE_TTL_SECONDS):
                return cached
            probe = self._probe_find_my_now(complete)
            self._find_my_probe = probe
            return probe

    def _probe_find_my_now(self, complete: bool) -> FindMyProbe:
        api = self._api
        if api is None:
            return FindMyProbe("error", "not signed in")
        try:
            root = api.get_webservice_url("findme") + "/fmipservice/client/web/"
            answer = self._find_my_request(api, root + "initClient", None)
            devices = {d["id"]: d for d in answer.get("content") or [] if isinstance(d, dict) and d.get("id")}
            polls = 0
            while complete and polls < self.FAMILY_POLLS and self._family_still_loading(answer):
                polls += 1
                time.sleep(self.FAMILY_POLL_DELAY_SECONDS)
                answer = self._find_my_request(api, root + "refreshClient", answer.get("serverContext"))
                devices.update({d["id"]: d for d in answer.get("content") or [] if isinstance(d, dict) and d.get("id")})
            fine(f"iCloud: Find My probe ok — {len(devices)} device(s)"
                 + (f" after {polls} family poll(s)" if polls else ""))
            return FindMyProbe("ok", f"{len(devices)} device(s)", list(devices.values()), complete=complete)
        except PyiCloudAuthRequiredException as e:
            info(f"iCloud: Find My probe — the session needs a fresh sign-in ({_brief(e)})")
            return FindMyProbe("reauth", _brief(e))
        except Exception as e:
            if self._is_reauthentication_error(e):
                info(f"iCloud: Find My probe — the session needs a fresh sign-in ({_brief(e)})")
                return FindMyProbe("reauth", _brief(e))
            error(f"iCloud: Find My probe failed: {_brief(e)}")
            return FindMyProbe("error", _brief(e))

    @staticmethod
    def _find_my_request(api: PyiCloudService, url: str, server_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        body: Dict[str, Any] = {"clientContext": dict(_FIND_MY_CLIENT_CONTEXT)}
        if server_context:
            body["serverContext"] = server_context
        answer = api.session.post(url, params=api.params, json=body).json()
        if not isinstance(answer, dict):
            raise iCloudAPIError(f"unexpected Find My answer: {type(answer).__name__}")
        return answer

    @staticmethod
    def _family_still_loading(answer: Dict[str, Any]) -> bool:
        user_info = answer.get("userInfo") or {}
        members = user_info.get("membersInfo") or {}
        return bool(user_info.get("hasMembers")) and any(
            isinstance(m, dict) and m.get("deviceFetchStatus") == "LOADING" for m in members.values())

    def _sign_in_for_find_my(self, why: str) -> Optional[str]:
        """One deliberate password sign-in on the live session because Find My rejected its
        cookies (Apple expires them within the hour), run only when a command needs Find My.
        The trust token normally spares the 2FA; a challenge Apple raises anyway is handed to
        the auto-2FA flow, never left open. Returns why Find My still cannot be used, None when
        the sign-in went through. Parked sign-ins are not attempted. Commands that want one at
        the same time share it: a caller that waited for another's sign-in gets its outcome."""
        seen = self._find_my_signins
        with self._find_my_signin_lock:
            if self._find_my_signins != seen:
                info(f"iCloud: Find My wanted a fresh sign-in ({why}), and another command ran one meanwhile "
                     f"— taking its outcome ({self._find_my_signin_outcome or 'signed in'})")
                return self._find_my_signin_outcome
            outcome = self._sign_in_for_find_my_now(why)
            self._find_my_signins += 1
            self._find_my_signin_outcome = outcome
            return outcome

    def _sign_in_for_find_my_now(self, why: str) -> Optional[str]:
        parked = self._parked_reason()
        if parked:
            info(f"iCloud: Find My wants a fresh sign-in ({why}), but sign-ins are parked — {parked}")
            return f"sign-ins are parked after Apple refused one: {parked}"
        api = self._api
        info(f"iCloud: Find My wants a fresh sign-in ({why}) — re-authenticating")
        started_at = time.time()
        trust_token_held = bool(api.session.data.get("trust_token"))
        self._signin_sms_wanted = True
        try:
            api.authenticate(force_refresh=True)
        except (PyiCloudFailedLoginException, PyiCloudAcceptTermsException) as e:
            self._auth_error = _brief(e)
            error(f"iCloud: re-authentication for Find My failed: {_brief(e)}")
            self._park(e)
            self._find_my_probe = FindMyProbe("error", _brief(e))
            return _brief(e)
        except PyiCloudConnectionException as e:
            error(f"iCloud: re-authentication for Find My failed: {_brief(e)}")
            self._find_my_probe = FindMyProbe("error", _brief(e))
            return _brief(e)
        finally:
            self._signin_sms_wanted = False
        self._auth_error = None
        self._find_my_probe = None
        api._devices = None                       # the device manager of the cookies Find My rejected
        self._challenge_from_fresh_signin = self._challenge_open()
        self._arm_auto_sent_code(started_at)
        if self._challenge_from_fresh_signin:
            info("iCloud: the re-authentication ended in a 2FA challenge"
                 + (" although the session held a trust token — Apple no longer honours it"
                    if trust_token_held else " — the session held no trust token")
                 + " — the auto-2FA flow completes it")
            self._find_my_probe = FindMyProbe("reauth", "2FA pending")
            return self.SESSION_EXPIRED_2FA_MESSAGE
        return None

    def requires_2fa(self) -> bool:
        """A 2FA challenge is open: one of our own password sign-ins ended on it and no code
        has been accepted since. pyicloud's flag alone is not it — it is also set on a
        token-restored session Apple calls untrusted, which has no challenge a code could
        answer and serves Find My and the setup endpoints as it is (Photos' CloudKit, which
        needs the trust, asks for the sign-in that opens a challenge when it breaks). Until
        2026-10-16 the verdict came from a Find My probe, a request Apple counted as a
        failed authentication every time the cookies had expired."""
        try:
            self._ensure_authenticated()
        except iCloudAuthenticationError:
            return False
        api = self._api
        if not api or not api.requires_2fa:
            return False
        if not self._challenge_from_fresh_signin:
            trace("iCloud: pyicloud flags the session untrusted, but no challenge is open — nothing to verify")
            return False
        return True

    def request_sms_2fa(self) -> str:
        """Request a 2FA code via SMS. Returns the obfuscated phone number.
        Must be called before send_2fa_code() so the SMS validation path is used."""
        if not self._api:
            raise iCloudAuthenticationError("Not authenticated")

        phone_verification = self._api._auth_data.get("phoneNumberVerification", {})
        phone = phone_verification.get("trustedPhoneNumber")
        if not phone:
            trace("iCloud: auth data carries no phone challenge — this happens when the last "
                  "authentication completed via session token instead of a password signin; "
                  f"auth_data keys: {sorted(self._api._auth_data.keys())}")
            raise iCloudAuthenticationError("No trusted phone number available for SMS")

        from pyicloud.const import CONTENT_TYPE_JSON, CONTENT_TYPE_TEXT
        headers = self._api._get_auth_headers(
            {"Accept": f"{CONTENT_TYPE_JSON}, {CONTENT_TYPE_TEXT}"}
        )
        # Capture before the PUT so a code that lands while the request is still
        # in flight isn't rejected as predating the request; assign only on
        # success so a failed PUT doesn't leave a phantom pending request.
        requested_at = time.time()
        resp = self._api.session.put(
            "https://idmsa.apple.com/appleauth/auth/verify/phone",
            json={"phoneNumber": self._phone_payload(phone), "mode": "sms"},
            headers=headers,
        )
        if resp.status_code != 200:
            raise iCloudAuthenticationError(f"SMS request failed (HTTP {resp.status_code})")

        self._api._auth_data["mode"] = "sms"
        self._api._auth_data["trustedPhoneNumber"] = phone
        # pyicloud 2.5.0: set delivery method so validate_2fa_code uses the SMS
        # path instead of the trusted device bridge path
        self._api._two_factor_delivery_method = "sms"

        obfuscated = phone.get("obfuscatedNumber", "???")
        self._sms_requested_at = requested_at
        info(f"iCloud: SMS 2FA code requested, sent to {obfuscated}")
        return obfuscated

    @staticmethod
    def _phone_payload(phone: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"id": phone["id"]}
        if isinstance(phone.get("nonFTEU"), bool):
            payload["nonFTEU"] = phone["nonFTEU"]
        return payload

    def sms_request_pending(self) -> bool:
        """True while a requested SMS code is still expected to arrive.
        The TUI uses this to avoid firing a second request (which would
        invalidate the code already in flight) when the user gets impatient."""
        requested = self._sms_requested_at
        return requested is not None and time.time() - requested < self.SMS_CODE_WAIT_SECONDS

    def on_needs_2fa(self):
        """Called by IntegrationMonitor while the session needs 2FA — drive the
        hands-off flow: request an SMS code, which the iMessage poller then
        auto-submits when it lands."""
        if self.sms_request_pending():
            trace("iCloud: SMS 2FA code already requested, waiting for it to arrive via iMessage")
            return
        if self._sms_requested_at is not None:
            info("iCloud: requested SMS 2FA code never arrived, requesting a new one")
        self._auto_request_sms("session needs 2FA")

    def _fresh_password_signin(self):
        """Sign in from scratch with the account password, discarding the persisted
        session token and cookies first. Re-auth via the saved session token can
        succeed for basic web APIs while carrying no live MFA challenge (auth_data
        comes back empty), and Apple rejects any SMS code submitted against such a
        session with 409 'Authentication required for Account' — without ever
        evaluating the code. Only a password (SRP) signin raises a challenge that
        codes can actually validate against."""
        parked = self._parked_reason()
        if parked:
            info(f"iCloud: no fresh password signin — sign-ins are parked ({parked})")
            raise iCloudAuthenticationError(f"sign-ins parked after Apple refused one: {parked}")
        info("iCloud: fresh password signin — discarding persisted session state")
        self._api = None
        self._sms_requested_at = None
        api = self._new_service(authenticate=False)
        api.session.clear_persistence()
        started_at = time.time()
        self._signin_sms_wanted = True
        try:
            api.authenticate()
        except (PyiCloudFailedLoginException, PyiCloudAcceptTermsException) as e:
            self._auth_error = _brief(e)
            error(f"iCloud: fresh password signin failed: {_brief(e)}")
            self._park(e)
            raise iCloudAuthenticationError(f"Fresh signin failed: {e}") from e
        except PyiCloudConnectionException as e:
            self._auth_error = _brief(e)
            error(f"iCloud: fresh password signin failed: {_brief(e)}")
            raise iCloudAuthenticationError(f"Fresh signin failed: {e}") from e
        finally:
            self._signin_sms_wanted = False
        self._api = api
        self._auth_error = None
        self._challenge_from_fresh_signin = True
        info(f"iCloud: fresh password signin complete "
             f"(requires_2fa={api.requires_2fa}, trusted={api.is_trusted_session})")
        self._arm_auto_sent_code(started_at)

    def _recovery_failed(self, why: str):
        """Give up on the current auto-recovery: enter the failure cooldown (without
        it the monitor retries — SRP signin + alert — every 15 min for as long as
        Apple keeps refusing) and tell the user."""
        error(f"iCloud: auto-2FA recovery failed — {why}")
        self._auto_sms_last_failure_ts = time.time()
        # This recovery is over — give the next one (after the cooldown) its own
        # fresh-signin budget. A surviving counter would trip the cap check
        # before any work, re-fail, re-arm the cooldown, and alert every 2h
        # forever without ever attempting another signin.
        self._fresh_signins_this_recovery = 0
        system_note(self.name, f"iCloud auto-2FA could not recover: {why}. Auto-recovery paused for "
                               f"{self.AUTO_SMS_FAILURE_COOLDOWN_SECONDS // 3600}h. Manual attention "
                               "needed in the Keep TUI console (iCloud tab).")

    def _auto_request_sms(self, reason: str, fresh_signin: bool = False):
        """Request an SMS 2FA code without user interaction. Falls back to a fresh
        password signin (dropping the persisted session token) when the current
        session cannot produce a usable challenge — pass fresh_signin=True to go
        straight there, e.g. after Apple rejected a code with 'Authentication
        required for Account', which means codes on this session are doomed."""
        with self._auto_sms_lock:
            if self.sms_request_pending():
                trace("iCloud: SMS 2FA request already pending, not requesting another")
                return
            parked = self._parked_reason()
            if parked:
                info(f"iCloud: not auto-requesting SMS 2FA code — sign-ins are parked ({parked})")
                return
            if not self._reader_live():
                info(f"iCloud: not auto-requesting SMS 2FA code ({reason}) — no reader of the Mac's text "
                     "messages is live, a code would go unread; the user enters it (icloud_login_code or the "
                     "console's iCloud tab)")
                if self._challenge_open_now():
                    self._note_challenge(self._sms_requested_at or 0.0)
                return
            limit = self._auto_sms_limiter.check()
            if limit:
                info(f"iCloud: not auto-requesting SMS 2FA code ({limit}) — "
                     "request manually in the TUI iCloud tab (Ctrl-E) if needed")
                return
            since_failure = time.time() - self._auto_sms_last_failure_ts
            if since_failure < self.AUTO_SMS_FAILURE_COOLDOWN_SECONDS:
                info(f"iCloud: not auto-requesting SMS 2FA code — last recovery attempt "
                     f"failed {int(since_failure // 60)}m ago, in cooldown for another "
                     f"{int((self.AUTO_SMS_FAILURE_COOLDOWN_SECONDS - since_failure) // 60)}m "
                     "(a manual request in the TUI iCloud tab still works)")
                return
            if not fresh_signin and not self._challenge_from_fresh_signin:
                info("iCloud: current auth challenge did not come from a fresh password "
                     "signin — codes requested on it get rejected, going straight to "
                     "a fresh signin")
                fresh_signin = True
            if not fresh_signin:
                try:
                    self._ensure_authenticated()
                    if not self._api or not self._api.requires_2fa:
                        trace("iCloud: no 2FA challenge active, nothing to auto-request")
                        return
                    obfuscated = self.request_sms_2fa()
                except Exception as e:
                    info(f"iCloud: SMS request on existing session failed ({_brief(e)}), "
                         "falling back to a fresh password signin")
                    fresh_signin = True
            if fresh_signin:
                if self._fresh_signins_this_recovery >= self.MAX_FRESH_SIGNINS_PER_RECOVERY:
                    self._recovery_failed(
                        f"already did {self._fresh_signins_this_recovery} fresh password "
                        "signin(s) this recovery and the session still cannot validate a "
                        "code — not signing in again")
                    return
                try:
                    self._fresh_signins_this_recovery += 1
                    self._fresh_password_signin()
                    if not self._api or not self._api.requires_2fa:
                        info("iCloud: session no longer requires 2FA after fresh signin")
                        return
                    if self.sms_request_pending():
                        obfuscated = (self._auto_sent_phone() or {}).get("obfuscatedNumber", "???")
                        reason += ", using the code sent by the signin itself"
                    else:
                        obfuscated = self.request_sms_2fa()
                except Exception as retry_err:
                    self._recovery_failed(
                        f"SMS request failed even after a fresh password signin ({_brief(retry_err)})")
                    return
            self._auto_sms_limiter.record()
            self._auto_sms_last_failure_ts = 0.0
            info(f"iCloud: auto-requested SMS 2FA code ({reason}), sent to {obfuscated} — "
                 "will auto-submit when it arrives via iMessage")

    def submit_user_code(self, code: str) -> str:
        """The code the user read off their phone and told the assistant (icloud_login_code):
        validated like the tab's Ctrl-E entry, outside the auto-submit's single-shot bookkeeping.
        Returns the sentence for the LLM."""
        digits = "".join(ch for ch in code if ch.isdigit())
        if len(digits) != 6:
            info(f"icloud_login_code: refused '{code}' — not a 6-digit code")
            return "iCloud refused the code: Apple's codes are 6 digits."
        if not self._challenge_open_now():
            status = self.get_auth_status()
            info(f"icloud_login_code: no 2FA challenge open — iCloud is {status['status']}")
            return f"iCloud is not waiting for a code — it is {status['status']}: {status['message']}"
        if self.send_2fa_code(digits):
            self._sms_requested_at = None
            info("icloud_login_code: Apple accepted the user's code")
            from assistant.integration.IntegrationMonitor import IntegrationMonitor
            IntegrationMonitor.get_instance().check_now(self.name)
            return "iCloud signed in — Apple accepted the code."
        info("icloud_login_code: Apple rejected the user's code")
        return ("Apple rejected the code. A code is single-use and lasts a few minutes; the newest text "
                "on the phone holds the current one — ask for it, or restart the sign-in with icloud_unlocked.")

    def send_2fa_code(self, code: str) -> bool:
        """
        Send 2FA verification code.
        Returns True if verification succeeded.
        """
        api = self._api
        if not api:
            raise iCloudAuthenticationError("Not authenticated")

        self._stale_session_rejection = False
        trace(f"iCloud: validating 2FA code, delivery_method={api.two_factor_delivery_method}")
        if api.two_factor_delivery_method == "sms":
            result = self._validate_sms_code_verbose(api, code)
        else:
            try:
                result = api.validate_2fa_code(code)
            except Exception as e:
                error(f"iCloud: 2FA validation exception: {_brief(e)}")
                return False
            trace(f"iCloud: validate_2fa_code returned {result}")
        if result:
            self.state = IntegrationState.OK
            self._sms_requested_at = None
            self._fresh_signins_this_recovery = 0
            self._find_my_probe = None       # "2FA pending" predates the code; the next listing decides
            self._challenge_answered.set()
            info("iCloud: 2FA verification successful")
            if not api.is_trusted_session:
                info("iCloud: Trusting session...")
                api.trust_session()
        else:
            error("iCloud: 2FA verification failed "
                  f"(requires_2fa={api.requires_2fa}, requires_2sa={api.requires_2sa}, "
                  f"is_trusted_session={api.is_trusted_session})")
        return result

    def _validate_sms_code_verbose(self, api: PyiCloudService, code: str) -> bool:
        """Mirror of pyicloud 2.5.0's validate_2fa_code() SMS branch, with the
        diagnostics the library swallows: validate_2fa_code() catches the HTTP
        error and returns a bare False, which cannot distinguish 'Apple rejected
        the code' from 'code accepted but the trust step failed' — and the
        recovery for those two states is opposite (a rejected code needs a new
        SMS, a failed trust must NOT request one against the consumed challenge)."""
        try:
            api._validate_sms_code(code)
        except PyiCloudAPIResponseException as e:
            body = (e.response.text or "")[:2000] if e.response is not None else ""
            if self._code_accepted_despite_status(e):
                # Apple answers an ACCEPTED code on this account's SMS flow with
                # HTTP 409 (the challenge form echoed back, no service_errors) and
                # hands out the new session token in the response headers. pyicloud
                # reads that as a failure and never trusts the session; the
                # 2026-08-30 probe showed 2sv/trust succeeding right after it.
                info(f"iCloud: Apple returned HTTP {e.code} to the SMS 2FA code but issued a "
                     "new session token — the code was accepted, continuing to trust")
            else:
                # A real rejection carries service_errors (-21669 'Incorrect
                # Verification Code') or comes without a session token: the code
                # was wrong, superseded, or the challenge behind it is gone.
                self._stale_session_rejection = (
                    e.code == 409
                    and "authentication required" in str(e.reason or "").lower())
                error(f"iCloud: Apple rejected the SMS 2FA code — "
                      f"reason={e.reason!r} code={e.code} "
                      f"stale_session={self._stale_session_rejection} body={body}")
                return False
        except Exception as e:
            error(f"iCloud: SMS 2FA code submission failed: {_brief(e)}")
            return False
        finally:
            api._clear_trusted_device_bridge_state()

        info("iCloud: SMS 2FA code accepted by Apple, trusting session...")
        trusted = api.trust_session()
        if not trusted:
            error("iCloud: 2FA code was ACCEPTED but the session trust step failed — "
                  "the challenge is consumed, so requesting another SMS code will not help")
        return not api.requires_2sa

    @staticmethod
    def _code_accepted_despite_status(e: PyiCloudAPIResponseException) -> bool:
        """True when a non-2xx answer to /verify/phone/securitycode is Apple's
        'accepted, carry on' (new X-Apple-Session-Token, no service_errors) rather
        than a rejection."""
        resp = e.response
        token = resp.headers.get("X-Apple-Session-Token") if resp is not None else None
        if not isinstance(token, str) or not token:
            return False
        try:
            data = resp.json()
        except ValueError:
            return True
        return not (isinstance(data, dict) and (data.get("hasError") or data.get("service_errors")))

    def submit_auto_2fa_code(self, code: str, received_at: Optional[float] = None):
        """Auto-submit a 2FA code detected from iMessage. Called from iMessage polling thread.

        Only codes received after the most recent request_sms_2fa() call are accepted,
        and only one code is submitted per request — Apple invalidates outstanding codes
        when a new one is requested, and a failed attempt can sour the whole challenge,
        so guessing with stale codes makes the legit code fail too."""
        if not self._api or not self._api.requires_2fa:
            trace("iCloud: ignoring auto-detected 2FA code — not awaiting 2FA")
            return
        requested_at = self._sms_requested_at
        if requested_at is None:
            info("iCloud: ignoring auto-detected 2FA code — no SMS request pending "
                 "(code was triggered elsewhere, e.g. a signin auto-send or another device)")
            return
        if received_at is not None and received_at < requested_at:
            info(f"iCloud: ignoring auto-detected 2FA code — it arrived {requested_at - received_at:.0f}s "
                 "before the current SMS request, so it belongs to an earlier attempt")
            return
        self._sms_requested_at = None  # single-shot: one submission per SMS request
        info(f"iCloud: auto-detected 2FA code from iMessage, submitting...")
        try:
            result = self.send_2fa_code(code)
            if result:
                self._auto_2fa_message = "2FA code auto-detected from iMessage — verified!"
                from assistant.integration.IntegrationMonitor import IntegrationMonitor
                IntegrationMonitor.get_instance().check_now("iCloud")
                self._announce_verified()
            else:
                self._auto_2fa_message = "2FA code auto-detected but verification failed"
                if self._stale_session_rejection and self._challenge_from_fresh_signin:
                    self._stale_session_rejection = False
                    # The 409 hit a challenge our own password signin raised seconds
                    # ago, so 'stale session' is the wrong diagnosis and repeating
                    # the recipe cannot help — on 2026-08-24 it looped four times
                    # (4 SRP signins + 4 SMS in 63s) until the SMS limiter tripped.
                    self._recovery_failed(
                        "Apple answered 409 'Authentication required for Account' to a code "
                        "on a challenge from a fresh password signin — retrying would loop")
                elif self._stale_session_rejection:
                    self._stale_session_rejection = False
                    # The session is dead, not the code — another code on this
                    # session is equally doomed, so recover with a from-scratch
                    # password signin and request the next code on that challenge.
                    system_note(self.name, "iCloud 2FA code auto-detected from iMessage but Apple rejected it "
                                           "on a stale session — signing in again and requesting a fresh code.")
                    self._auto_request_sms("code rejected on a stale session",
                                           fresh_signin=True)
                else:
                    # The code was stale (superseded by a newer one) or the challenge is
                    # soured — request a fresh code so the flow recovers on its own.
                    system_note(self.name, "iCloud 2FA code auto-detected from iMessage but Apple rejected it — "
                                           "requesting a fresh code.")
                    self._auto_request_sms("previous code was rejected")
        except Exception as e:
            error(f"iCloud: auto 2FA submission failed: {_brief(e)}")
            self._auto_2fa_message = f"2FA auto-submit error: {e}"

    def _announce_verified(self):
        """The verified code reaches the user as the turn that opens the conversation while
        setup is in progress — the user is waiting on this sign-in, as on the Full Disk Access
        grant — and as a system row otherwise: a re-auth outside the setup is routine. The
        health check ran first, so the turn sees iCloud signed in."""
        from assistant import FirstRun
        from assistant.Notifier import notifier
        if not FirstRun.in_ftue(verbose=False):
            trace("iCloud: setup is not in progress — the verified code is a system note")
        elif notifier().open_conversation(VERIFIED_TURN):
            info("iCloud: the conversation is opened with the verified-code turn")
            return
        else:
            info("iCloud: no chat took the verified-code turn — a system note carries it")
        system_note(self.name, VERIFIED_NOTE)

    @property
    def auto_2fa_message(self) -> Optional[str]:
        """Return and clear the one-shot auto-2FA status message for the TUI."""
        msg = self._auto_2fa_message
        self._auto_2fa_message = None
        return msg

    def get_auth_status(self) -> Dict[str, Any]:
        """Current authentication status for TUI display. No request of its own: the Find My
        verdict shown is the last listing's (the tab's device table runs one, cached)."""
        parked = self._parked_reason()
        if parked:
            return {
                "status": "parked",
                "message": f"Sign-ins parked after Apple refused one: {parked} — Ctrl-R once the account works again",
            }

        if self._auth_error:
            return {
                "status": "error",
                "message": self._auth_error,
            }

        if self._api is None:
            return {
                "status": "not_initialized",
                "message": "Not connected to iCloud",
            }

        if self.requires_2fa():
            return {
                "status": "needs_2fa",
                "message": "2FA verification required",
            }

        # A "reauth" verdict stands until a sign-in or an accepted code drops it: expired
        # cookies do not come back on their own, so asking Find My again before then would be
        # one more refused request every TTL for nothing. A transient error may expire.
        probe = self._find_my_probe
        if probe and probe.verdict == "error" and time.time() - probe.at >= self.FIND_MY_PROBE_TTL_SECONDS:
            probe = None
        if probe and probe.verdict == "reauth":
            # The resting state after a restart or half an hour idle: Find My's cookies are
            # gone, the session is fine, and the next device command signs in by itself.
            return {
                "status": "needs_reauth",
                "message": "Signed in — Find My is idle and signs in when a device is asked for",
            }
        if probe and probe.verdict == "error":
            return {
                "status": "error",
                "message": f"Find My: {probe.detail}",
            }

        # Session works for basic APIs but pyicloud still flags requires_2fa —
        # CloudKit needs full 2FA completion to work.
        if self._api.requires_2fa:
            return {
                "status": "partial_auth",
                "message": "Connected — login not fully verified; auto-2FA will fix via SMS if photos break",
            }

        return {
            "status": "authenticated",
            "message": "Connected to iCloud",
        }

    def get_devices_list(self) -> str:
        """Return formatted list of configured device display names for prompt ("" when none)."""
        specs = self._device_specs
        return ", ".join(spec.label() for spec in specs)

    def configured_device_count(self) -> int:
        """How many icloud.devices entries exist — the index the next one gets."""
        return len(self._device_specs)

    def get_icloud_devices(self, sign_in: bool = False) -> List[Dict[str, Any]]:
        """Every device the iCloud account sees (not the config), each with `configured_as`
        (the display name it is configured under, or None) and `slot` (its index in
        icloud.devices, or None) — read through the Find My probe, so a listing never wakes a
        device. When Find My wants a fresh sign-in, `sign_in` (the icloud_devices command)
        does it; the TUI's table, which asks every half minute, does not — Ctrl-R is its way."""
        try:
            self._ensure_authenticated()
        except iCloudAuthenticationError:
            return []
        if not self._api or self.requires_2fa():
            info("iCloud: not listing devices — the session needs 2FA")
            return []
        probe = self._probe_find_my(complete=True)
        if probe.verdict == "reauth" and sign_in and not self._sign_in_for_find_my("device listing"):
            probe = self._probe_find_my(force=True, complete=True)
        if probe.verdict != "ok":
            info(f"iCloud: not listing devices — Find My probe {probe.verdict}: {probe.detail}")
            return []
        specs = self._device_specs
        devices = []
        for data in probe.devices:
            slot, display_name = next(((i, spec.display_name) for i, spec in enumerate(specs)
                                       if spec.matches(data)), (None, None))
            devices.append({
                "id": data.get("id", ""),
                "name": data.get("name", "Unknown"),
                "model": data.get("deviceDisplayName", ""),
                "class": data.get("deviceClass", ""),
                "configured_as": display_name,
                "slot": slot,
            })
        fine(f"iCloud: account sees {len(devices)} device(s), {len(specs)} configured")
        return devices

    @staticmethod
    def _is_reauthentication_error(exc: Exception) -> bool:
        if isinstance(exc, PyiCloudAuthRequiredException):
            return True
        error_str = str(exc).lower()
        return "re-authentication required" in error_str or ("authentication" in error_str and "required" in error_str)

    def _ensure_authenticated_and_ready(self):
        """Signed in with no 2FA challenge open, or the reason as an iCloudAuthenticationError.
        A session with an open challenge is kept as it is — the auto-2FA flow is completing it,
        and dropping it (as this did until 2026-10-16) meant the next call opened another."""
        self._ensure_authenticated()
        if self._api and self.requires_2fa():
            self.state = IntegrationState.NEEDS_2FA
            info("iCloud: a 2FA challenge is open — not using Find My until the code is accepted")
            raise iCloudAuthenticationError(self.SESSION_EXPIRED_2FA_MESSAGE)

    def _challenge_wait_pointless(self) -> Optional[str]:
        """Why waiting for the open challenge to be answered would be a wait for nothing — None
        while a code is on its way or the auto-2FA flow may still request one."""
        if self.sms_request_pending():
            return None
        if not self._reader_live():
            return "nothing on this Mac reads the code Apple texts, the user enters it"
        parked = self._parked_reason()
        if parked:
            return f"sign-ins are parked: {parked}"
        if time.time() - self._auto_sms_last_failure_ts < self.AUTO_SMS_FAILURE_COOLDOWN_SECONDS:
            return "the auto-2FA flow is in its failure cooldown"
        return None

    def _wait_for_find_my(self, purpose: str):
        """`_ensure_authenticated_and_ready`, but a 2FA challenge the auto-2FA flow is completing
        is waited out (up to CHALLENGE_WAIT_SECONDS) instead of failing the command: a ring asked
        while the session re-signs in then happens a minute later instead of never. Any other
        reason Find My cannot be used is raised at once, as is a challenge nothing will answer."""
        deadline = None
        while True:
            try:
                self._ensure_authenticated_and_ready()
                return
            except iCloudAuthenticationError:
                if not self._challenge_open_now():
                    raise
                pointless = self._challenge_wait_pointless()
                if pointless:
                    info(f"iCloud: {purpose} is not waiting for the 2FA code — {pointless}")
                    raise
                if deadline is None:
                    deadline = time.time() + self.CHALLENGE_WAIT_SECONDS
                remaining = deadline - time.time()
                if remaining <= 0:
                    info(f"iCloud: {purpose} waited {self.CHALLENGE_WAIT_SECONDS}s for the 2FA code — giving up")
                    raise iCloudAuthenticationError(
                        f"iCloud is still waiting for its 2FA code after {self.CHALLENGE_WAIT_SECONDS}s "
                        "(manual entry: icloud_login_code or the console's iCloud tab)")
                info(f"iCloud: {purpose} waits for the 2FA code to be accepted ({remaining:.0f}s left)")
                self._challenge_answered.wait(min(remaining, 5.0))

    def _find_my_operation(self, operation, context: str):
        """Run a Find My operation once the session is ready for it, waiting out a 2FA challenge
        before it — and the one the re-sign-in inside `_run_with_reauthentication` may open."""
        self._wait_for_find_my(context)
        try:
            return self._run_with_reauthentication(operation, context)
        except iCloudAuthenticationError:
            if not self._challenge_open_now():
                raise
            self._wait_for_find_my(context)
            return operation()

    def _devices_now(self):
        """pyicloud's device manager, rebuilt: the one from the last command holds that
        command's records, and no monitor thread refreshes them (refresh_interval=None)."""
        api = self._api
        api._devices = None
        return api.devices

    @staticmethod
    def _find_target_device(spec: iCloudDeviceSpec, devices):
        target_device = next((d for d in devices if spec.matches(d.data)), None)
        if not target_device:
            available = [f"{d.data.get('name')} ({d.data.get('deviceDisplayName') or d.data.get('deviceClass') or '?'}, "
                         f"id {(d.data.get('id') or '')[:8]}…)" for d in devices]
            info(f"iCloud: configured device '{spec.label()}' (id {spec.id[:8]}…) is not among the "
                 f"{len(available)} the account sees — its id may have changed; icloud_devices() shows the current ones")
            raise iCloudAPIError(
                f"Device '{spec.label()}' (id={spec.id}) not found in iCloud. Available: {', '.join(available)}"
            )
        return target_device

    def _run_with_reauthentication(self, operation, failure_context: str):
        """Run a Find My operation; when Find My rejects the session's cookies, sign in once —
        deliberately, on the live session (pyicloud's own re-auth is disabled, see
        _refresh_find_my_without_reauth) — and run it again."""
        try:
            return operation()
        except iCloudAPIError:
            raise
        except Exception as e:
            if not self._is_reauthentication_error(e):
                raise
            info(f"iCloud session expired {failure_context}, re-authenticating...")
            trouble = self._sign_in_for_find_my(failure_context)
            if trouble:
                raise iCloudAuthenticationError(f"Find My is not available: {trouble}") from e
            return operation()

    def _play_sound_for_spec(self, spec: iCloudDeviceSpec) -> str:
        target_device = self._find_target_device(spec, self._devices_now())
        target_device.play_sound()
        return target_device.data.get("name") or spec.label()

    def play_sound(self, device_name: Optional[str]) -> str:
        """Play sound on the specified device (Find My iPhone)."""
        info("Playing sound on iCloud device...")
        spec = self._select_device_spec(device_name)
        played_device_name = self._find_my_operation(lambda: self._play_sound_for_spec(spec), "the ring")
        note(f"Playing sound on {played_device_name}")

        return f"Playing sound on {spec.label()}"

    def _wait_for_fresh_location(self, spec: iCloudDeviceSpec):
        """Return the device for spec, triggering an active Find My locate and
        polling until it reports a fresh fix (isOld is False) or the timeout
        elapses. pyicloud's AppleDevice.location is a cached property and the
        background monitor thread only pulls last-known positions, so without
        this we'd report stale locations (e.g. "home" long after they left)."""
        deadline = time.time() + self._LOCATE_TIMEOUT_SECONDS
        attempts = 0
        devices = self._devices_now()
        while True:
            target_device = self._find_target_device(spec, devices)
            location = target_device.location
            if not location or not bool(location.get("isOld")):
                if attempts:
                    info(f"iCloud: got a fresh fix for {spec.label()} after {attempts} locate attempt(s)")
                return target_device
            if time.time() >= deadline:
                info(
                    f"iCloud: {spec.label()} location still stale after {attempts} locate attempt(s) "
                    f"({self._LOCATE_TIMEOUT_SECONDS:.0f}s budget) — falling back to last-known fix"
                )
                return target_device
            attempts += 1
            trace(f"iCloud: {spec.label()} fix is stale, requesting active locate (attempt {attempts})")
            try:
                devices.refresh(locate=True)
            except Exception as e:
                if self._is_reauthentication_error(e):
                    raise
                info(f"iCloud: active locate refresh failed for {spec.label()}: {_brief(e)} — falling back to last-known fix")
                return target_device
            time.sleep(self._LOCATE_POLL_INTERVAL_SECONDS)

    def _location_for_spec(self, spec: iCloudDeviceSpec) -> str:
        """Inner operation: read location for a resolved device spec.
        Wrapped by _run_with_reauthentication so an expired session retries once."""
        target_device = self._wait_for_fresh_location(spec)
        location = target_device.location
        if not location:
            note(f"iCloud: location unavailable for {spec.label()}")
            return f"Location unavailable for {spec.label()} (device offline, location services off, or no recent fix)."

        latitude = location.get("latitude")
        longitude = location.get("longitude")
        accuracy = location.get("horizontalAccuracy")
        timestamp_ms = location.get("timeStamp")
        is_old = bool(location.get("isOld"))
        is_inaccurate = bool(location.get("isInaccurate"))

        if latitude is None or longitude is None:
            note(f"iCloud: location dict for {spec.label()} missing coordinates: {location}")
            return f"Location unavailable for {spec.label()} (no coordinates in response)."

        trace(f"iCloud {spec.label()}: lat={latitude} lon={longitude} accuracy={accuracy}m old={is_old} inaccurate={is_inaccurate}")

        place = reverse_geocode(latitude, longitude)
        if place:
            info(f"Resolved {spec.label()} to {place}")
        else:
            info(f"iCloud: reverse geocode returned no result for {spec.label()} ({latitude}, {longitude})")

        return self._format_location(
            spec.label(),
            place,
            latitude,
            longitude,
            accuracy,
            timestamp_ms,
            is_old,
            is_inaccurate,
        )

    @staticmethod
    def _format_location(
        label: str,
        place: Optional[str],
        latitude: float,
        longitude: float,
        accuracy: Optional[float],
        timestamp_ms: Optional[float],
        is_old: bool,
        is_inaccurate: bool,
    ) -> str:
        headline = f"{label}: {place}" if place else f"{label}: location unavailable as a place name"
        if is_old:
            # Lead with staleness so the assistant doesn't confidently report an
            # out-of-date position as the person's current whereabouts.
            headline += " (STALE — last-known location, the device has not reported a fresh fix)"
        lines = [headline]

        accuracy_str = f"±{int(round(accuracy))}m" if accuracy is not None else "accuracy unknown"
        lines.append(f"GPS: {latitude:.5f}, {longitude:.5f} ({accuracy_str})")

        if timestamp_ms is not None:
            try:
                age_seconds = max(0, time.time() - (float(timestamp_ms) / 1000.0))
                lines.append(f"Updated {ICloudIntegration._humanize_age(age_seconds)} ago")
            except (TypeError, ValueError):
                pass

        if is_inaccurate:
            lines.append("Note: fix is inaccurate.")

        return "\n".join(lines)

    @staticmethod
    def _humanize_age(age_seconds: float) -> str:
        if age_seconds < 60:
            return f"{int(age_seconds)} seconds"
        if age_seconds < 3600:
            return f"{int(age_seconds // 60)} minutes"
        if age_seconds < 86400:
            hours = age_seconds / 3600
            return f"{hours:.1f} hours" if hours < 10 else f"{int(hours)} hours"
        days = age_seconds / 86400
        return f"{days:.1f} days" if days < 10 else f"{int(days)} days"

    def get_location(self, device_name: Optional[str]) -> str:
        """Fetch location for the specified device via Find My iPhone."""
        info(f"Fetching iCloud location for device: {device_name!r}")
        spec = self._select_device_spec(device_name)
        return self._find_my_operation(lambda: self._location_for_spec(spec), "the location lookup")

    @property
    def api(self) -> Optional[PyiCloudService]:
        """Get the authenticated PyiCloudService instance, or None if unavailable."""
        try:
            self._ensure_authenticated()
            if self._api and not self.requires_2fa():
                return self._api
        except Exception:
            pass
        return None

    def refresh_connection(self, by_user: bool = False):
        """Sign in again on the session files (the token when it still serves, the password
        when Apple wants it). `by_user` — Ctrl-R in the iCloud tab, the icloud_unlocked
        command — is the user saying the account works again: it lifts the park and, when Find
        My wants more than the restored token, signs in for it too, now. Photos' refresh on an
        auth error is automatic: the park holds, Find My is left alone (CloudKit does not need
        it), and a session with an open 2FA challenge is kept for the auto-2FA flow to finish
        rather than replaced by a sign-in that opens another."""
        if by_user:
            self._lift_park()
        elif self._api and self._api.requires_2fa and self._challenge_from_fresh_signin:
            info("iCloud: refresh asked while a 2FA challenge is open — keeping the session, nudging the auto-2FA flow")
            self.on_needs_2fa()
            return
        self._api = None
        self._auth_error = None
        # New signin = new auth challenge; any SMS code in flight for the old
        # session is dead, so don't let the poller submit it.
        self._sms_requested_at = None
        self._ensure_authenticated()
        if by_user and self._api and not self.requires_2fa():
            probe = self._probe_find_my(force=True)
            if probe.verdict == "reauth" and not self._sign_in_for_find_my("manual refresh"):
                probe = self._probe_find_my(force=True)
            if probe.verdict == "ok":
                self.state = IntegrationState.OK
                info("iCloud connection refreshed, session is functional")
            else:
                info(f"iCloud connection refreshed, but Find My is not serving it: {probe.detail}")
        if self._api and self._api.requires_2fa:
            # Session may work for basic APIs, but the browser isn't trusted
            # (CloudKit/photos broken) — kick off the automatic SMS 2FA flow.
            self._auto_request_sms("session untrusted after refresh")


iCloudAPI = ICloudIntegration
