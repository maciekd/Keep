"""Sets up local AI on a new install: oMLX (github.com/jundot/omlx) serving one model on this Mac."""
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from common import Platform
from common.install.Terminal import GB, InstallFailed, Terminal, fmt_elapsed

# The one runner, version and model that were tested together; they move together, in a Keep release.
OMLX_VERSION = "0.8.3"
TAP = "jundot/omlx"
TAP_URL = "https://github.com/jundot/omlx"
# The tap is oMLX's own repository. This is the commit whose formula builds OMLX_VERSION — the
# release's own tag still carries the formula of the release before it.
TAP_COMMIT = "1cd5aaa9c75942560331deaa35f787123f7cf2b0"
FORMULA = f"{TAP}/omlx"
MODEL_REPO = "Jundot/Qwen3.8-27B-oQ4e-mtp"
MODEL_REVISION = "836759eefbef7ef588a5ffd44cc59a640808f944"   # the repository has been re-quantized in place before
MODEL = MODEL_REPO.split("/")[1]    # the folder's name is the id oMLX serves it under; "qwen" in it is what
                                    # makes Keep write Qwen's call syntax
MODEL_LABEL = "Qwen 3.8 27B"
DOWNLOAD_GB = 18

DEFAULT_PORT = 8000
# oMLX recomputes its memory ceiling from what the other apps hold and aborts a request above it.
# The model with a long prompt peaks at about 21 GB: under CEILING_BELOW_GB of memory the ceiling
# is fixed, so that macOS compresses the other apps instead (docs/plans/LOCAL_AI_INSTALL_PLAN.md).
CEILING_GB = 22
CEILING_BELOW_GB = 36

HOMEBREW_INSTALLER = "https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh"
BREW_PATHS = ("/opt/homebrew/bin/brew",)
BREW_ENV = {"HOMEBREW_NO_AUTO_UPDATE": "1", "HOMEBREW_NO_ENV_HINTS": "1"}   # an update would move the tap off TAP_COMMIT
START_WAIT_SECONDS = 120
LOAD_TIMEOUT_SECONDS = 900
WATCH_SECONDS = 0.5


@dataclass(frozen=True)
class LocalModel:
    """What config.yaml says of the local server: `base_url` and `api_key` only where they are not
    the defaults."""
    model: str
    base_url: str = ""
    api_key: str = ""


def find_brew() -> str:
    """Homebrew's `brew`, "" when there is none. By its own path too: a Homebrew installed a
    minute ago is on no PATH yet."""
    return shutil.which("brew") or next((path for path in BREW_PATHS if os.access(path, os.X_OK)), "")


class LocalRunner:
    """`set_up()` installs what is missing, starts the server and returns the model once it has
    answered; anything that stops it is an `InstallFailed` saying what. What is already there is
    the user's and is kept: an installed oMLX whatever its version, a downloaded model, every key
    of an existing settings.json. Every fact about oMLX is in this file."""

    def __init__(self, terminal: Terminal, env: dict, home: Path | None = None):
        self.terminal = terminal
        self.env = env
        self.home = Path(home) if home else Path.home() / ".omlx"
        self.sleep = time.sleep

    def set_up(self) -> LocalModel:
        brew = self.ensure_brew()
        installed = self.ensure_omlx(brew)
        changed = self.ensure_settings()
        downloaded = self.ensure_model()
        self.start_service(brew, restart=installed or changed or downloaded)
        self.prove()
        port = self.port()
        return LocalModel(MODEL, "" if port == DEFAULT_PORT else f"{self.address()}/v1", self.api_key())

    # ------------------------------------------------------------ Homebrew --

    def ensure_brew(self) -> str:
        t = self.terminal
        brew = find_brew()
        if brew:
            return brew
        if not t.interactive:
            raise InstallFailed("Homebrew is not installed, and there is no terminal for its installer to ask on.")
        t.note("Homebrew is not installed. Its own installer runs now and asks for your password:")
        t.note()
        t.log(f"--- Homebrew's installer ({HOMEBREW_INSTALLER})")
        status = self.install_brew()
        t.note()
        brew = find_brew()
        if status != 0 or not brew:
            raise InstallFailed(f"Homebrew's installer did not complete (exit {status}).")
        t.ok("Homebrew installed")
        return brew

    def install_brew(self) -> int:
        """Homebrew's installer, on the terminal itself: it asks and it prints its own progress."""
        try:
            script = requests.get(HOMEBREW_INSTALLER, timeout=30)
            script.raise_for_status()
        except requests.RequestException as e:
            raise InstallFailed(f"Homebrew's installer could not be fetched ({type(e).__name__}).")
        with tempfile.NamedTemporaryFile("w", suffix=".sh") as f, open(Terminal.TTY) as keyboard:
            f.write(script.text)
            f.flush()
            return subprocess.run(["/bin/bash", f.name], stdin=keyboard).returncode

    # ---------------------------------------------------------------- oMLX --

    def ensure_omlx(self, brew: str) -> bool:
        """True when it was installed now."""
        t = self.terminal
        status, versions = self.output([brew, "list", "--versions", "omlx"])
        if status == 0 and versions:
            t.ok(f"oMLX is already installed ({versions.split()[-1]}); kept as it is")
            return False
        env = {**os.environ, **BREW_ENV}
        self.command("Add oMLX to Homebrew", [brew, "tap", TAP, TAP_URL], env)
        status, repo = self.output([brew, "--repo", TAP])
        if status != 0 or not repo:
            raise InstallFailed("Homebrew does not say where it keeps the oMLX tap.")
        self.command(f"Choose oMLX {OMLX_VERSION}", ["git", "-C", repo, "checkout", "--quiet", TAP_COMMIT])
        self.command(f"Build oMLX {OMLX_VERSION} (from source: several minutes)", [brew, "install", FORMULA], env)
        if not t.run("Keep oMLX at this version (brew pin)", [brew, "pin", "omlx"], env):
            t.warn("oMLX is not pinned: a `brew upgrade` will move it to a version Keep was not tested with.")
        return True

    def command(self, label: str, argv: list, env: dict | None = None) -> None:
        if not self.terminal.run(label, argv, env):
            raise InstallFailed(f"{label} did not complete.")

    def output(self, argv: list) -> tuple[int, str]:
        """(exit status, what it printed) of a command that only answers."""
        try:
            done = subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
        except (OSError, subprocess.SubprocessError) as e:
            self.terminal.log(f"{' '.join(argv)}: {type(e).__name__}")
            return 127, ""
        return done.returncode, done.stdout.strip()

    # ------------------------------------------------------------ settings --

    def settings_path(self) -> Path:
        return self.home / "settings.json"

    def settings(self) -> dict:
        path = self.settings_path()
        if not path.is_file():
            return {}
        try:
            settings = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            raise InstallFailed(f"{path} cannot be read ({type(e).__name__}).", "It is oMLX's settings file and is left as it is.")
        if not isinstance(settings, dict):
            raise InstallFailed(f"{path} does not hold oMLX's settings.")
        return settings

    def section(self, name: str) -> dict:
        section = self.settings().get(name)
        return section if isinstance(section, dict) else {}

    def port(self) -> int:
        port = self.section("server").get("port")
        return port if isinstance(port, int) and port > 0 else DEFAULT_PORT

    def api_key(self) -> str:
        """An existing install's key; a new one has none, and oMLX on loopback asks for none."""
        key = self.section("auth").get("api_key")
        return key if isinstance(key, str) else ""

    def address(self) -> str:
        return f"http://127.0.0.1:{self.port()}"

    def models_dir(self) -> Path:
        model = self.section("model")
        dirs = model.get("model_dirs") or ([model["model_dir"]] if model.get("model_dir") else [])
        return Path(dirs[0]).expanduser() if dirs and isinstance(dirs[0], str) else self.home / "models"

    def ensure_settings(self) -> bool:
        """Writes the memory ceiling where the Mac needs one; True when the file changed. As few
        keys as possible — oMLX renames them between releases — and none that is already set."""
        t = self.terminal
        memory = Platform.memory_gb()
        if memory >= CEILING_BELOW_GB:
            t.log(f"settings: {memory} GB of memory, oMLX's own memory guard is kept")
            return False
        settings = self.settings()
        guard = settings.get("memory") if isinstance(settings.get("memory"), dict) else {}
        if "memory_guard_tier" in guard:
            if guard["memory_guard_tier"] != "custom":
                t.dim(f"oMLX's memory guard is set to {guard['memory_guard_tier']} in {self.settings_path()}; kept. "
                      f"If the model does not load, a fixed ceiling of {CEILING_GB} GB is what fits this Mac.")
            return False
        settings["memory"] = {**guard, "memory_guard_tier": "custom", "memory_guard_custom_ceiling_gb": CEILING_GB}
        self.home.mkdir(parents=True, exist_ok=True)
        self.settings_path().write_text(json.dumps(settings, indent=2) + "\n")
        t.ok(f"oMLX may use {CEILING_GB} GB of this Mac's {memory} GB")
        return True

    # ------------------------------------------------------------ the model --

    def ensure_model(self) -> bool:
        """True when anything was downloaded."""
        t = self.terminal
        target = self.models_dir() / MODEL
        files = self.files()
        total = sum(size for _, size in files)
        if all((target / name).is_file() and (target / name).stat().st_size == size for name, size in files):
            t.ok(f"{MODEL_LABEL} is already in {target}; kept")
            return False
        t.dim(f"{MODEL_LABEL} comes from Hugging Face ({MODEL_REPO}) into {target}.")
        t.dim("An interrupted download continues where it stopped when the installer runs again.")
        started = t.clock()
        failure = []

        def work():
            try:
                self.download(target)
            except Exception as e:
                failure.append(e)

        worker = threading.Thread(target=work, daemon=True, name="model-download")
        bar = t.bar(MODEL_LABEL, total)
        worker.start()
        try:
            while worker.is_alive():
                bar.update(self.on_disk(target))
                worker.join(WATCH_SECONDS)
        finally:
            bar.close()
        if failure:
            t.log(f"model download: {type(failure[0]).__name__}: {failure[0]}")
            raise InstallFailed(f"The download of {MODEL_LABEL} did not complete ({type(failure[0]).__name__}).",
                                "What arrived is kept: running the installer again continues it.")
        t.ok(f"{MODEL_LABEL} downloaded: {total / GB:.1f} GB {t.styled('dim', '(' + fmt_elapsed(t.clock() - started) + ')')}")
        return True

    def files(self) -> list[tuple[str, int]]:
        """(name, bytes) of every file of the model at MODEL_REVISION."""
        from huggingface_hub import HfApi
        from huggingface_hub.utils import logging as hub_logging
        hub_logging.set_verbosity_error()     # its advice about tokens is not the installer's to print
        try:
            info = HfApi().model_info(MODEL_REPO, revision=MODEL_REVISION, files_metadata=True)
        except Exception as e:
            self.terminal.log(f"model files: {type(e).__name__}: {e}")
            raise InstallFailed(f"Hugging Face did not answer for {MODEL_REPO} ({type(e).__name__}).")
        return [(file.rfilename, file.size or 0) for file in info.siblings or []]

    def download(self, target: Path) -> None:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()     # the bar is ours: one for the model, not one per file
        snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_dir=str(target))

    def downloaded_gb(self) -> int:
        """What is here of the model already, in whole GB: a download that was interrupted, or all of it."""
        return self.on_disk(self.models_dir() / MODEL) // GB

    @staticmethod
    def on_disk(target: Path) -> int:
        """The bytes under *target*, the files still arriving among them."""
        total = 0
        for folder, _, names in os.walk(target):
            for name in names:
                try:
                    total += os.lstat(os.path.join(folder, name)).st_size
                except OSError:
                    pass        # a part file renamed between the listing and the reading
        return total

    # --------------------------------------------------------- the service --

    def start_service(self, brew: str, restart: bool) -> None:
        """oMLX as Homebrew's launchd service: started at login and revived. One that runs outside
        the service keeps the port, and a second would only die on it again and again."""
        t = self.terminal
        if self.models() is not None and not self.service_runs(brew):
            t.warn(f"oMLX already answers on port {self.port()}, started by hand: left running, so nothing starts it at login.")
            t.note(f"After stopping it: brew services start {FORMULA}")
            return
        self.command("Start oMLX (from now on at every login)",
                     [brew, "services", "restart" if restart else "start", FORMULA], {**os.environ, **BREW_ENV})

    def service_runs(self, brew: str) -> bool:
        status, said = self.output([brew, "services", "info", "omlx", "--json"])
        try:
            return status == 0 and bool(json.loads(said)[0].get("running"))
        except (ValueError, LookupError, AttributeError, TypeError):
            return False

    # ------------------------------------------------------------ the proof --

    def headers(self) -> dict:
        key = self.api_key()
        return {"Authorization": f"Bearer {key}"} if key else {}

    def models(self) -> list[str] | None:
        """The ids the server on the port lists; None when nothing answers there."""
        try:
            response = requests.get(f"{self.address()}/v1/models", headers=self.headers(), timeout=5)
            if response.status_code in (401, 403):
                raise InstallFailed(f"What answers on port {self.port()} wants an API key that {self.settings_path()} does not hold.")
            return [str(model.get("id")) for model in response.json().get("data", [])]
        except requests.RequestException:
            return None
        except (ValueError, AttributeError, TypeError):
            return []

    def complete(self) -> tuple[int, str]:
        """(HTTP status, the server's own words on a failure) of a one-token completion — which
        loads the model, and so shows that it fits."""
        request = {"model": MODEL, "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}
        try:
            response = requests.post(f"{self.address()}/v1/chat/completions", headers=self.headers(), json=request,
                                     timeout=LOAD_TIMEOUT_SECONDS)
        except requests.RequestException as e:
            return 0, type(e).__name__
        return response.status_code, "" if response.status_code == 200 else response.text[:300].strip()

    def prove(self) -> None:
        t = self.terminal
        port = self.port()

        def listed() -> list[str] | None:
            deadline = t.clock() + START_WAIT_SECONDS
            while True:
                ids = self.models()
                if (ids and MODEL in ids) or t.clock() >= deadline:
                    return ids
                self.sleep(1)

        ids = t.wait("Waiting for oMLX to answer", listed)
        if ids is None:
            raise InstallFailed(f"oMLX does not answer on port {port}.", "Its log: $(brew --prefix)/var/log/omlx.log")
        if MODEL not in ids:
            raise InstallFailed(f"What answers on port {port} does not serve {MODEL}.",
                                "It lists: " + (", ".join(ids) or "nothing"))
        t.ok(f"oMLX serves {MODEL} on port {port}")
        status, said = t.wait(f"Loading {MODEL_LABEL} into memory for the first time", self.complete)
        if status != 200:
            t.log(f"first completion: HTTP {status} {said}")
            raise InstallFailed(f"{MODEL_LABEL} did not answer (HTTP {status or 'none'}{': ' + said if said else ''}).",
                                "It may not fit beside what else runs on this Mac.")
        t.ok(f"{MODEL_LABEL} answers")
