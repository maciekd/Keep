"""The HTTP client of a Philips Hue bridge: transport only. It carries the CLIP v2 resource
reads and writes and the pairing request that earns the application key; finding the bridge is
HueDiscovery's job and what a light or a scene is, is HueModel's. The bridge serves a
certificate signed by Signify's own CA with the bridge id as its hostname — one macOS rejects
outright, not merely as untrusted — so every call to it goes through `SystemTrust.unverified_context()`:
plain OpenSSL, verifying nothing. The bridge is on the LAN, reached by its address."""
import requests
import urllib3
from requests.adapters import HTTPAdapter

from common import SystemTrust
from common.logging import info, trace

API_PATH = "/clip/v2/resource"
PAIR_PATH = "/api"
DEVICE_TYPE = "keep#server"             # what the bridge lists the key under in the Hue app
TIMEOUT = 10
LINK_BUTTON_ERROR = 101                 # the bridge's error type for "link button not pressed"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class HueError(Exception):
    """The bridge answered with an error: the message is what the LLM should read."""


class LinkButtonNotPressed(HueError):
    """A pairing request while the bridge's link button has not been pressed."""


class _BridgeAdapter(HTTPAdapter):
    """Hands urllib3 the unverified context; `session.verify = False` beside it keeps requests
    from asking for a CA bundle the context does not have."""

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = SystemTrust.unverified_context()
        return super().init_poolmanager(*args, **kwargs)


def bridge_session() -> requests.Session:
    """A session that reaches the bridge: TLS without verification, on a plain OpenSSL context."""
    session = requests.Session()
    session.verify = False
    session.mount("https://", _BridgeAdapter())
    return session


class HueClient:
    def __init__(self, host: str, app_key: str):
        self.host = host.strip()
        self.base = f"https://{self.host}"
        self._session = bridge_session()
        self._session.headers["hue-application-key"] = app_key

    def get(self, rtype: str) -> list[dict]:
        """Every resource of *rtype* (`light`, `room`, `scene`, ...)."""
        response = self._session.get(f"{self.base}{API_PATH}/{rtype}", timeout=TIMEOUT)
        data = self._data(response)
        trace(f"Philips Hue: GET {rtype} → {len(data)} resources")
        return data

    def put(self, rtype: str, rid: str, body: dict) -> list[dict]:
        response = self._session.put(f"{self.base}{API_PATH}/{rtype}/{rid}", json=body, timeout=TIMEOUT)
        return self._data(response)

    @staticmethod
    def _data(response: requests.Response) -> list[dict]:
        """The `data` list of a CLIP v2 answer; its `errors` become a HueError, a non-JSON
        failure an HTTPError."""
        try:
            payload = response.json()
        except ValueError:
            response.raise_for_status()
            raise HueError(f"the bridge answered HTTP {response.status_code} without JSON")
        errors = payload.get("errors") if isinstance(payload, dict) else None
        if errors:
            raise HueError("; ".join(str(e.get("description") or "error") for e in errors))
        response.raise_for_status()
        return payload.get("data") or []


def pair(host: str) -> str:
    """The application key the bridge at *host* hands out while its link button is pressed:
    LinkButtonNotPressed when it was not, HueError for any other refusal."""
    host = host.strip()
    response = bridge_session().post(f"https://{host}{PAIR_PATH}", json={"devicetype": DEVICE_TYPE, "generateclientkey": True},
                                     timeout=TIMEOUT)
    response.raise_for_status()
    answers = response.json()
    for answer in answers if isinstance(answers, list) else [answers]:
        success = answer.get("success") or {}
        if success.get("username"):
            info(f"Philips Hue: paired with the bridge at {host}")
            return success["username"]
        error = answer.get("error") or {}
        if error.get("type") == LINK_BUTTON_ERROR:
            info(f"Philips Hue: the bridge at {host} wants its link button pressed first")
            raise LinkButtonNotPressed(error.get("description") or "link button not pressed")
        if error:
            raise HueError(error.get("description") or "pairing refused")
    raise HueError(f"unexpected pairing answer: {answers}")

