"""The loopback-by-paste half of an OAuth code flow, shared by every integration whose account signs in from the chat.

A desktop or loopback OAuth client may only redirect to an address on the machine the browser
runs on, which is not the server. So the integration hands the user an authorization URL, the
browser lands on a `http://localhost…/?code=…&state=…` page that never loads, and the user
pastes that whole address back. `SignIn` is what `begin()` hands out — the URL and the state
the answer must carry, with the instruction text every chat shows — and `pasted_code` turns
the pasted answer into the code to exchange, or says why there is none: the whole address is
required (a bare code would skip the state check), and anything without `code=` and `state=`
— the bare redirect, the sign-in link itself, a sentence — is refused as such instead of being
sent to the token endpoint as if it were a code."""
from dataclasses import dataclass
from typing import Optional
from urllib.parse import parse_qs, urlsplit

from common.logging import info

WHOLE_ADDRESS_HINT = "paste the whole address the browser landed on"


@dataclass(frozen=True)
class SignIn:
    """What the user needs — the URL to open — and the state the pasted answer must carry.
    `address` is the account the user should sign in as, when the integration knows it."""
    url: str
    state: str
    redirect_uri: str
    address: str = ""

    @property
    def instructions(self) -> str:
        who = f" as {self.address}" if self.address else ""
        return (f"open {self.url} in any browser and sign in{who}; the browser then lands on a "
                f"{self.redirect_uri}… page that will not load — that is expected — copy that page's whole address "
                f"from the address bar and paste it back")


def parse_pasted(pasted: str) -> tuple[str, str, str]:
    """(code, state, error) out of what the user pasted: the whole redirected URL or its query
    string. Anything without those parameters parses to nothing."""
    text = pasted.strip().strip("\"'`<>").strip()
    if "code=" not in text and "state=" not in text and "error=" not in text:
        return "", "", ""
    query = urlsplit(text).query if "?" in text else text.lstrip("?")
    values = parse_qs(query, keep_blank_values=True)
    first = lambda key: (values.get(key) or [""])[0].strip()
    return first("code"), first("state"), first("error")


def pasted_code(pasted: str, expected_state: Optional[str], service: str, login: str) -> tuple[str, str]:
    """The code to exchange out of a pasted answer, or ("", why not). *expected_state* is the
    flow's state, None while no flow is in progress; *service* names the provider in the
    refusal, *login* the command that starts a flow (`google_login("Personal")`)."""
    code, state, refusal = parse_pasted(pasted)
    if refusal:
        info(f"{service}: sign-in refused by {service}: {refusal}")
        return "", f"{service} refused the sign-in: {refusal}"
    if expected_state is None:
        info(f"{service}: an answer was pasted while no sign-in is in progress")
        return "", f"no sign-in is in progress — run {login} first"
    if not code or not state:
        info(f"{service}: the paste carries {'no code' if not code else 'no state'}, nothing exchanged")
        return "", f"no code and state in what was pasted — {WHOLE_ADDRESS_HINT}"
    if state != expected_state:
        info(f"{service}: sign-in pasted with a state that is not this flow's")
        return "", f"the pasted address belongs to another sign-in — run {login} again and paste that one"
    return code, ""
