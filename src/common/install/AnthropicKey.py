"""The Anthropic API key of a new install: asked for, checked with Anthropic, and the workspace it
acts in when the key is not tied to one."""
import json

import requests

from common.install.Terminal import InstallFailed, Terminal

KEYS_URL = "https://platform.claude.com/settings/keys"
WORKSPACES_URL = "https://platform.claude.com/settings/workspaces"
BILLING_URL = "https://platform.claude.com/settings/billing"
MESSAGES_URL = "https://api.anthropic.com/v1/messages"
CHECK_MODEL = "claude-haiku-5-5"

OK, CREDIT, WORKSPACE, BAD_WORKSPACE = "200", "credit", "workspace", "bad-workspace"
REJECTED = ("401", "403")
KEY_ATTEMPTS = 3          # keys Anthropic was asked about
WORKSPACE_ATTEMPTS = 3
SLIPS = 10                # empty or malformed entries in a row: nobody is typing
ENV_KEY, ENV_WORKSPACE = "ANTHROPIC_API_KEY", "ANTHROPIC_WORKSPACE_ID"


class AnthropicKey:
    """`obtain()` returns the key and the workspace id ("" for a key tied to one workspace). The
    key comes from `ANTHROPIC_API_KEY` or is typed; either way Anthropic is asked about it. A key
    that cannot be checked right now is kept, one that is rejected is asked for again."""

    def __init__(self, terminal: Terminal, env: dict, keep_dir: str):
        self.terminal = terminal
        self.env = env
        self.keep_dir = keep_dir
        self.said = ""          # Anthropic's own words about the last key checked

    def obtain(self) -> tuple[str, str]:
        t = self.terminal
        from_env = self.env.get(ENV_KEY, "")
        key = from_env
        workspace = self.env.get(ENV_WORKSPACE, "")
        if key:
            t.dim(f"Using {ENV_KEY} from the environment.")
        else:
            self._explain_key()
        attempt = workspace_attempt = slips = 0
        while True:
            asked = False
            if not key:
                if attempt >= KEY_ATTEMPTS:
                    raise InstallFailed(f"No valid API key after {KEY_ATTEMPTS} attempts.", self._rerun_hint())
                if slips >= SLIPS:
                    raise InstallFailed("No API key entered.", self._rerun_hint())
                key = "".join(t.ask("Anthropic API key:", secret=True).split())
                asked = True
            if not key.startswith("sk-ant-"):
                t.warn("That does not look like an Anthropic key (they start with sk-ant-)." if key else "Nothing entered.")
                key = ""
                slips += 1
                continue
            slips = 0
            attempt += asked
            code = self.verify(key, workspace)
            if code == OK:
                t.ok("Key verified with the Anthropic API" + (f" (workspace {workspace})" if workspace else ""))
                return key, workspace
            if code == CREDIT:
                if self._wait_for_credit(bool(from_env)):
                    continue
                return key, workspace
            if code in (WORKSPACE, BAD_WORKSPACE):
                if code == WORKSPACE:
                    self._explain_workspace()
                else:
                    t.warn(f"Anthropic does not know workspace {workspace}, or this key cannot use it. "
                           f"Check the ID column at {WORKSPACES_URL}.")
                workspace_attempt += 1
                if workspace_attempt > WORKSPACE_ATTEMPTS:
                    raise InstallFailed(
                        f"No usable workspace id after {WORKSPACE_ATTEMPTS} attempts.",
                        f"Create a key with Scope set to Default workspace at {KEYS_URL} and re-run the installer, "
                        f"or pass {ENV_WORKSPACE}=wrkspc_... in the environment.")
                workspace = "".join(t.ask("Workspace id (or press Enter to paste a new key instead):").split())
                if workspace.startswith("wrkspc_"):
                    continue
                if workspace:
                    t.warn("That does not look like a workspace id (they start with wrkspc_).")
                    workspace = ""
                    continue
                if from_env:
                    raise InstallFailed(
                        f"{ENV_KEY} from the environment is not tied to a workspace.",
                        f"Pass {ENV_WORKSPACE}=wrkspc_... as well, or create a key with Scope set to Default workspace at {KEYS_URL}.")
                key = ""
                continue
            if code not in REJECTED:
                t.warn(f"Could not verify the key right now (HTTP {code or 'none'}{': ' + self.said if self.said else ''}); "
                       "keeping it, the server will tell you if it is wrong.")
                return key, workspace
            t.warn(f"Anthropic rejected that key (HTTP {code}). Check for a typo or create a new one.")
            key = ""
            if from_env:
                raise InstallFailed(f"{ENV_KEY} from the environment was rejected.")

    def verify(self, key: str, workspace: str) -> str:
        """What Anthropic made of the key: the HTTP status as text ("" when there was no answer),
        or WORKSPACE when the key is not tied to one workspace and Anthropic wants to be told which,
        BAD_WORKSPACE when the id given is not one the key can act in, CREDIT when the key is good
        and its organization has no credit. Anthropic's own words are left in `said`."""
        code, body = self.post(key, workspace)
        self.said = self._message(body)
        self.terminal.log(f"key check: HTTP {code or 'none'}" + (f" - {self.said}" if self.said else ""))
        said = self.said.lower()
        if code == "400":
            if "anthropic-workspace-id" in said and "required" in said.split("anthropic-workspace-id", 1)[1]:
                return WORKSPACE
            if "credit balance" in said:
                return CREDIT
            if "workspace" in said and workspace:
                return BAD_WORKSPACE
        if code == "404" and "workspace" in said and workspace:
            return BAD_WORKSPACE
        return code

    def post(self, key: str, workspace: str) -> tuple[str, str]:
        """(status, body) of a real one-token message on Haiku, a fraction of a cent: the free
        count_tokens endpoint is not documented to check the balance, and an unfunded key that
        passes here leaves the first chat silent."""
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        if workspace:
            headers["anthropic-workspace-id"] = workspace
        message = {"model": CHECK_MODEL, "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}
        try:
            response = requests.post(MESSAGES_URL, headers=headers, json=message, timeout=20)
        except requests.RequestException as e:
            self.terminal.log(f"key check: no answer ({type(e).__name__})")
            return "", ""
        return str(response.status_code), response.text

    @staticmethod
    def _message(body: str) -> str:
        try:
            message = json.loads(body)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            return ""
        return message if isinstance(message, str) else ""

    def _rerun_hint(self) -> str:
        return f"Re-run the installer when you have one, or pass it as {ENV_KEY}=... in the environment."

    def _explain_key(self) -> None:
        t = self.terminal
        t.note("Keep runs on Claude. An API key is the only way to use it without breaking Anthropic's terms of service.")
        t.note(f"Create one at {t.bold(KEYS_URL)} (it starts with sk-ant-). In the dialog:")
        t.note("  Expires: Never")
        t.note("  Scope:   Default workspace")
        t.note(f"The key is stored only in {self.keep_dir}/config.yaml. Typing is hidden.")
        t.note()
        t.note("Keep uses your organization's API credit. On a Claude Max, Ultra or Team plan you get")
        t.note("$100–500 of it a month for free: claim it at claude.ai → Settings → Billing → API credits")
        t.note("(link the Console organization you create this key in). Otherwise add credits at")
        t.note(f"{t.bold(BILLING_URL)}.")
        t.note()

    def _explain_workspace(self) -> None:
        t = self.terminal
        t.warn("This key's Scope is Organization, so Anthropic needs to be told which workspace to use.")
        t.note(f"The easy fix: create a new key at {t.bold(KEYS_URL)} with Scope set to")
        t.note("Default workspace (and Expires to Never). Then paste the new key here.")
        t.note("Or keep this key and paste a workspace id instead: it starts with wrkspc_ and is in the")
        t.note(f"ID column at {t.bold(WORKSPACES_URL)}.")
        t.note()

    def _wait_for_credit(self, from_env: bool) -> bool:
        """The key is good and its organization has no credit: waits while the user claims or buys
        some. True to check again, False to go on without credit."""
        t = self.terminal
        t.warn("The key works, but its organization has no API credit yet.")
        t.note("On a Claude Max, Ultra or Team plan, claim the plan's monthly credit at claude.ai → Settings →")
        t.note("Billing → API credits and link this key's Console organization (a new subscription can claim")
        t.note(f"it after 7 days on the plan). Otherwise add credits at {t.bold(BILLING_URL)}.")
        t.note()
        if from_env:
            t.warn(f"Keeping {ENV_KEY} from the environment. Keep cannot answer until the organization has credit.")
            return False
        answer = t.ask("Press Enter to check again, or type skip to finish without credit:")
        if "".join(answer.split()).lower() == "skip":
            t.warn("Keeping the key. Keep cannot answer until the organization has credit.")
            return False
        t.dim("Checking again...")
        return True
