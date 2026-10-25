"""The Graph v1.0 calls the mail and calendar accounts share: a bearer token on every request,
paging, and the two answers that need a rule — 401 means the token is gone, 429 is retried once
after the Retry-After the server names. A request whose connection Graph drops before answering
(RemoteDisconnected, a reset) is sent once more on a new connection, unless it is a POST, which
may already have been carried out.

Everything else that goes wrong becomes a GraphError with the short message Graph sends,
which the account turns into the error string the LLM reads."""
import time
from typing import Optional

import requests

from common.logging import error, info, trace
from integrations.microsoft.MicrosoftAuth import MicrosoftAuth, NotSignedIn, not_signed_in_hint

BASE_URL = "https://graph.microsoft.com/v1.0"
TIMEOUT = 30
MAX_RETRY_AFTER = 60
RETRIED_ON_DROP = {"GET", "PATCH", "DELETE"}


class GraphError(Exception):
    """Graph answered with an error status; `status` carries it."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class GraphClient:
    def __init__(self, auth: MicrosoftAuth):
        self._auth = auth

    @property
    def name(self) -> str:
        return self._auth.name

    def get(self, path: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> dict:
        return self._call("GET", path, params=params, headers=headers)

    def get_all(self, path: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> list:
        """Every item of a paged collection, following `@odata.nextLink` to the end."""
        items: list = []
        page = self.get(path, params, headers)
        items.extend(page.get("value", []))
        pages = 1
        while page.get("@odata.nextLink"):
            page = self._call("GET", page["@odata.nextLink"], headers=headers)
            items.extend(page.get("value", []))
            pages += 1
        if pages > 1:
            trace(f"{self.name}: {len(items)} items over {pages} pages of {path}")
        return items

    def post(self, path: str, json: dict) -> dict:
        return self._call("POST", path, json=json)

    def patch(self, path: str, json: dict) -> dict:
        return self._call("PATCH", path, json=json)

    def delete(self, path: str) -> None:
        self._call("DELETE", path)

    def _call(self, method: str, path: str, params=None, headers=None, json=None, retried: bool = False) -> dict:
        url = path if path.startswith("https://") else BASE_URL + path
        token = self._auth.token()
        sent = {"Authorization": f"Bearer {token}", "Accept": "application/json", **(headers or {})}
        started = time.time()
        response = self._send(method, url, path, params, sent, json)
        trace(f"{self.name}: {method} {path.split('?')[0]} -> {response.status_code} in {time.time() - started:.1f}s")
        if response.status_code == 401:
            info(f"{self.name}: Graph answered 401 — the token is not accepted")
            raise NotSignedIn(f"token not accepted, {not_signed_in_hint(self.name)}")
        if response.status_code == 429:
            if retried:
                raise GraphError(429, "Graph is throttling the account, try again in a minute")
            wait = min(MAX_RETRY_AFTER, max(1, int(response.headers.get("Retry-After") or 5)))
            info(f"{self.name}: Graph answered 429, waiting {wait}s and retrying once")
            time.sleep(wait)
            return self._call(method, path, params, headers, json, retried=True)
        if response.status_code >= 400:
            message = self._message(response)
            error(f"{self.name}: Graph {method} {path.split('?')[0]} failed: {response.status_code} {message}")
            raise GraphError(response.status_code, message)
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()

    def _send(self, method: str, url: str, path: str, params, headers: dict, json):
        try:
            return requests.request(method, url, params=params, headers=headers, json=json, timeout=TIMEOUT)
        except requests.ConnectionError as e:
            if method not in RETRIED_ON_DROP:
                raise
            info(f"{self.name}: {method} {path.split('?')[0]} lost its connection ({e}), retrying once on a new one")
        return requests.request(method, url, params=params, headers=headers, json=json, timeout=TIMEOUT)

    @staticmethod
    def _message(response) -> str:
        try:
            body = response.json()
            err = body.get("error") or {}
            return str(err.get("message") or err.get("code") or body)[:300]
        except ValueError:
            return (response.text or f"HTTP {response.status_code}")[:300]
