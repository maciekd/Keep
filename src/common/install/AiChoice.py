"""Where a new install thinks — Claude or a model on this Mac: the gate and the question."""
import shutil
import textwrap
from pathlib import Path

from common import Platform
from common.install import LocalRunner
from common.install.AnthropicKey import ENV_KEY
from common.install.Terminal import InstallFailed, Terminal

CLOUD, LOCAL = "cloud", "local"
ENV_CHOICE = "KEEP_AI"
MIN_MACOS = 15            # the oMLX formula's own floor
MIN_MEMORY_GB = 32        # the model and a long prompt take about 21 GB
MIN_FREE_GB = 30          # the model, oMLX and its disk cache; what is here of the model already counts
ATTEMPTS = 3
MANUAL, NOT_ASKED = "manual", "not_asked"      # what the install report says beside CLOUD and LOCAL
ASK_KEEP = "start on Claude and ask Keep to move you to local AI"


class AiChoice:
    """`choose()` returns CLOUD or LOCAL. LOCAL is what the installer sets up itself, and it is
    offered only on a Mac that can carry it. Without a person to ask — no terminal, or a key
    already in the environment — the answer is CLOUD, unless `KEEP_AI` says otherwise."""

    def __init__(self, terminal: Terminal, env: dict):
        self.terminal = terminal
        self.env = env
        self.wanted = env.get(ENV_CHOICE, "").strip().lower()
        self.lacks = ""             # for the install report: the requirement not met, as one word
        self.answer = NOT_ASKED     # and what was answered: cloud, local or manual; env_… from KEEP_AI

    @property
    def explicit(self) -> bool:
        """Local was asked for by name: a failure of it is then the install's, not a reason to
        go on with Claude unasked."""
        return self.wanted == LOCAL

    def choose(self) -> str:
        t = self.terminal
        if self.wanted not in ("", CLOUD, LOCAL):
            raise InstallFailed(f"{ENV_CHOICE}={self.env[ENV_CHOICE]} is neither {LOCAL} nor {CLOUD}.")
        missing = self.missing()
        if self.wanted == LOCAL:
            if missing:
                raise InstallFailed(f"{ENV_CHOICE}={LOCAL}, but {missing}.",
                                    f"For your own local model, {ASK_KEEP}: it sets it up with you.")
            t.dim(f"{ENV_CHOICE}={LOCAL}: setting up {LocalRunner.MODEL_LABEL} on this Mac.")
            self.answer = f"env_{LOCAL}"
            return LOCAL
        if self.wanted == CLOUD:
            self.answer = f"env_{CLOUD}"
            return CLOUD
        if missing:
            t.dim(f"Local AI is not set up by the installer here: {missing}.")
            t.dim(f"To run Keep on a local model, {ASK_KEEP}.")
            return CLOUD
        if self.env.get(ENV_KEY) or not t.interactive:
            return CLOUD
        return self.ask()

    def missing(self) -> str:
        """The first requirement this machine does not meet, in words; "" when it meets them all."""
        self.lacks = ""
        if not Platform.is_mac():
            self.lacks = "mac"
            return "it sets one up on a Mac only"
        if not Platform.is_apple_silicon():
            self.lacks = "apple_silicon"
            return "this Mac has an Intel chip, and the model needs Apple Silicon"
        version, memory, free = Platform.macos_version(), Platform.memory_gb(), self.free_gb()
        have = min(self.downloaded_gb(), LocalRunner.DOWNLOAD_GB)
        if version < MIN_MACOS:
            self.lacks = "macos"
            return f"this Mac runs macOS {version or '?'}, and local AI needs macOS {MIN_MACOS}"
        if memory < MIN_MEMORY_GB:
            self.lacks = "memory"
            return f"this Mac has {memory or '?'} GB of memory, and local AI needs {MIN_MEMORY_GB} GB"
        if free < MIN_FREE_GB - have:
            self.lacks = "disk"
            return f"the disk has {free} GB free, and local AI needs {MIN_FREE_GB - have} GB"
        return ""

    def downloaded_gb(self) -> int:
        """What is here of the model already. An oMLX whose settings cannot be read counts as
        none: that is the local setup's failure to report, not one of an install on Claude."""
        try:
            return self.runner().downloaded_gb()
        except InstallFailed as e:
            self.terminal.log(f"AI choice: what is downloaded of the model is unknown ({e}), counting none")
            return 0

    def runner(self) -> "LocalRunner.LocalRunner":
        return LocalRunner.LocalRunner(self.terminal, self.env)

    @staticmethod
    def free_gb() -> int:
        return shutil.disk_usage(Path.home()).free // 2 ** 30

    def ask(self) -> str:
        t = self.terminal
        local = f"{LocalRunner.MODEL_LABEL} on this Mac: an {LocalRunner.DOWNLOAD_GB} GB download, 100% local AI"
        if not LocalRunner.find_brew():
            local += ". Installs Homebrew, which asks for your password"
        t.note("Keep can think in the cloud or on this Mac.")
        t.note()
        self._option("1", "Claude (Anthropic)",
                     "The most capable; secrets go in the Keep Vault, your requests go through Anthropic servers")
        self._option("2", "Local, automatic setup", local)
        self._option("3", "Local, manual setup", "Set up your own local LLM model/server (advanced)")
        t.note()
        for attempt in range(1, ATTEMPTS + 1):
            answer = "".join(t.ask("Choice [1]:").split())
            if answer in ("", "1"):
                self.answer = CLOUD
                return CLOUD
            if answer == "2":
                self.answer = LOCAL
                return LOCAL
            if answer == "3":
                self.answer = MANUAL
                t.dim(f"For your own model or server, {ASK_KEEP}: it sets it up with you.")
                return CLOUD
            if attempt < ATTEMPTS:
                t.warn("Please answer 1, 2 or 3.")
        t.warn(f"No 1, 2 or 3 after {ATTEMPTS} attempts: going on with Claude.")
        self.answer = CLOUD
        return CLOUD

    def _option(self, number: str, name: str, description: str) -> None:
        """The name on a line, the description under it, dimmed and wrapped to the terminal: beside
        the name it would break mid-line on a fresh Terminal window's 80 columns."""
        t = self.terminal
        t.note(f"  {t.bold(number)}  {name}")
        for line in textwrap.wrap(description, max(30, min(t.columns(), 100) - 11)):
            t.note(f"     {t.styled('dim', line)}")
