"""Writes the config.yaml of a new install — a Keep server's or an iMessage satellite's."""
import contextlib
import io
import secrets
import string
from pathlib import Path

import yaml

from common import PrivateFile
from common.install.Terminal import InstallFailed, Terminal

SATELLITE_ROLE = "imessage_satellite"
SATELLITE_PORT = 8898     # not the server's 8888: a second user shares the Mac's ports
SECRET_LENGTH = 32
YES, NO = ("1", "true", "yes", "on"), ("0", "false", "no", "off")
LAN_ATTEMPTS = 3


def scalar(value) -> str:
    """*value* as YAML writes it on one line, quoted where it has to be."""
    return yaml.safe_dump(value, default_flow_style=True, width=10 ** 6).split("\n")[0]


class InstallConfig:
    """The file is written as the lines a person will read and edit — every key on a line of its
    own from a fixed column, with its comment — since the scripts read it with awk
    (`scripts/role.sh`) and the owner with an editor. The values go through a YAML library."""

    def __init__(self, terminal: Terminal, env: dict):
        self.terminal = terminal
        self.env = env
        self.path = Path(env["KEEP_DIR"]) / "config.yaml"
        self.from_master = env.get("KEEP_INSTALL_BRANCH") == "master"

    def write_server(self, api_key: str, workspace: str) -> None:
        lines = [
            f"# Keep configuration, written by install.sh on {self.terminal.today()}. config.example.yaml documents",
            "# every setting; the assistant sets up the optional integrations itself once it runs.",
            "api_keys:",
            f"  anthropic: {scalar(api_key)}",
        ]
        if workspace:
            lines.append(f"  anthropic_workspace: {scalar(workspace)}   "
                         "# the key is not tied to one workspace, so every request names this one")
        lines += [
            "developer_release: true   # installed from master, the development line; restart_server keeps following it"
            if self.from_master else
            "developer_release: false  # restart_server follows the release branch; true: master, the development line",
            "vault:",
            "  # Encrypts vault.bin, the secrets the assistant stores for you. Lose it and the vault is",
            "  # lost; change it and the existing vault.bin can no longer be opened.",
            f"  key: {self._secret()}",
            "telemetry:",
            "  # Beta usage telemetry; config.example.yaml says what goes out. enabled: false turns it off.",
            *self._telemetry(),
        ]
        self._write(lines)
        self.terminal.note("A random vault passphrase was generated for you. It lives in config.yaml: back that file up.")

    def write_satellite(self) -> None:
        """No API key and no vault: nothing on a satellite uses either. The file has to be there
        before the first start — without one `ConfigManager` writes the server's skeleton."""
        token = self._secret()
        lan_access = self.lan_access()
        self._write([
            f"# Keep configuration of an iMessage satellite, written by install.sh on {self.terminal.today()}. No assistant",
            "# runs on this install: it serves this Mac user's Messages to a Keep server, which is given the",
            "# address and the token below — as imessage.host and imessage.host_token when the account is the",
            "# owner's own, as imessage_chat.host and imessage_chat.host_token when it is the assistant's own",
            "# Apple ID. config.example.yaml documents the keys.",
            f"role: {SATELLITE_ROLE}",
            "imessage_satellite:",
            f"  port: {SATELLITE_PORT}",
            "  # What the Keep server must present. A secret: whoever holds it and reaches the port reads and",
            "  # sends this account's messages. A new value here must be entered on the server as well.",
            f"  token: {token}",
            "  # false: only this Mac reaches the satellite (the Keep server is another user of this Mac).",
            "  # true: the network does (the server is on another Mac); the token and the messages then cross",
            "  # it unencrypted, so only on a network you trust. Read at start.",
            f"  lan_access: {scalar(lan_access)}",
            "# Where the install came from. A running satellite follows the Keep server that calls it: it",
            "# moves itself to that server's line (release or master) and version.",
            "developer_release: true   # installed from master, the development line"
            if self.from_master else "developer_release: false  # installed from the release branch",
            "# Nobody sits at a satellite and no assistant is there to restart it, so launchd starts it again",
            "# within 5 minutes of dying. Applied at each start: to keep a stopped satellite down, set this to",
            "# false and run start.sh once more, then stop.sh.",
            "watchdog: true",
            "telemetry:",
            "  # The installer's own events (started, finished, or the step it failed in); a running satellite",
            "  # sends nothing. config.example.yaml says what goes out. enabled: false turns it off.",
            *self._telemetry(),
        ])

    def lan_access(self) -> bool:
        """Whether the satellite answers the network: KEEP_LAN_ACCESS, else one question, else —
        with no terminal to ask on — this Mac only, the answer that opens nothing."""
        t = self.terminal
        wanted = self.env.get("KEEP_LAN_ACCESS", "")
        if wanted.lower() in YES:
            t.dim("KEEP_LAN_ACCESS is set: the satellite will answer the network (a Keep server on another Mac).")
            return True
        if wanted.lower() in NO:
            t.dim("KEEP_LAN_ACCESS is off: the satellite will answer this Mac only (a Keep server under another user here).")
            return False
        if wanted:
            raise InstallFailed(f"KEEP_LAN_ACCESS={wanted} is neither 1 nor 0.")
        if not t.interactive:
            t.dim("No terminal to ask on: the satellite will answer this Mac only (imessage_satellite.lan_access: false).")
            t.dim("For a Keep server on another Mac, set it to true in config.yaml and run start.sh again.")
            return False
        t.note("Where does your Keep server run?")
        t.note("  1  on this Mac, under another user: the satellite answers this Mac only")
        t.note("  2  on another Mac: the satellite answers the network, and the token and the messages")
        t.note("     cross it unencrypted — only on a network you trust")
        for attempt in range(1, LAN_ATTEMPTS + 1):
            answer = "".join(t.ask("1 or 2 (Enter for 1):").split())
            if answer in ("", "1"):
                t.ok("This Mac only (lan_access: false)")
                return False
            if answer == "2":
                t.ok("The network (lan_access: true)")
                return True
            if attempt < LAN_ATTEMPTS:
                t.warn("Please answer 1 or 2.")
        t.warn(f"No 1 or 2 after {LAN_ATTEMPTS} attempts: this Mac only. imessage_satellite.lan_access in config.yaml changes it.")
        return False

    def _telemetry(self) -> list[str]:
        if self.env.get("KEEP_NO_TELEMETRY"):
            return ["  enabled: false"]
        return ["  enabled: true", f"  id: {scalar(self.env.get('KEEP_INSTALL_TELEMETRY_ID', ''))}"]

    @staticmethod
    def _secret() -> str:
        alphabet = string.ascii_letters + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(SECRET_LENGTH))

    def _write(self, lines: list[str]) -> None:
        said = io.StringIO()
        with contextlib.redirect_stdout(said):      # PrivateFile traces to the console; here that is the log's
            PrivateFile.write(self.path, "\n".join(lines) + "\n")
        self.terminal.log(f"config: wrote {self.path} (0600)")
        self.terminal.ok("Wrote config.yaml (readable by you only)")
