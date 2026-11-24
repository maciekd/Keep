"""The installer's lines, questions and log, as `install.sh` draws them."""
import os
import sys
import termios
from datetime import datetime

from common import Platform


class InstallFailed(Exception):
    """The install cannot go on: *reason* is the line the user reads (and the telemetry's), *notes*
    the lines under it that say what to do."""

    def __init__(self, reason: str, *notes: str):
        super().__init__(reason)
        self.reason = reason
        self.notes = notes


class Terminal:
    """What the installer prints and asks, in the look of `install.sh` (`ok`, `warn`, `dim`, `note`,
    `step`). A question goes to the terminal the installer was started from and never to stdin,
    which is the pipe from curl; everything said is also in the install log."""

    TTY = "/dev/tty"

    def __init__(self, log_path: str = "", step: int = 0, steps: int = 0, out=None, env=None):
        env = os.environ if env is None else env
        self.out = out or sys.stdout
        self.log_path = log_path
        self.step_number = step
        self.steps = steps
        self.step_name = ""
        colour = self.out.isatty() and not env.get("NO_COLOR") and env.get("TERM", "dumb") != "dumb"
        codes = {"bold": "1", "dim": "2", "green": "32", "red": "31", "yellow": "33", "cyan": "36", "reset": "0"}
        self._codes = {name: f"\033[{code}m" if colour else "" for name, code in codes.items()}
        self.interactive = self._has_terminal()

    def _has_terminal(self) -> bool:
        """Told by opening it: /dev/tty exists for a process that has none (ssh without -t,
        launchd), and only the open fails there."""
        try:
            os.close(os.open(self.TTY, os.O_RDWR))
            return True
        except OSError:
            return False

    def styled(self, style: str, text: str) -> str:
        return f"{self._codes[style]}{text}{self._codes['reset']}"

    def bold(self, text: str) -> str:
        return self.styled("bold", text)

    def _print(self, text: str = "") -> None:
        print(text, file=self.out, flush=True)

    def log(self, text: str) -> None:
        if not self.log_path:
            return
        with open(self.log_path, "a") as f:
            f.write(text + "\n")

    def step(self, name: str) -> None:
        self.step_number += 1
        self.step_name = name
        self._print(f"\n  {self.styled('cyan', f'[{self.step_number}/{self.steps}]')} {self.bold(name)}")
        self.log("")
        self.log(f"=== [{self.step_number}/{self.steps}] {name}")

    def ok(self, text: str) -> None:
        self._print(f"  {self.styled('green', '✔')} {text}")
        self.log(f"ok: {text}")

    def warn(self, text: str) -> None:
        self._print(f"  {self.styled('yellow', '!')} {text}")
        self.log(f"warn: {text}")

    def dim(self, text: str) -> None:
        self._print(f"  {self.styled('dim', text)}")
        self.log(text)

    def note(self, text: str = "") -> None:
        """A line of running text; shown, not logged (the satellite's token is one)."""
        self._print(f"    {text}" if text else "")

    def failed(self, failure: InstallFailed) -> None:
        """The failure as the user reads it. The log's last two lines are what `install.sh` reports
        it by: the step, then the reason."""
        self._print(f"\n  {self.styled('red', '✖ ' + failure.reason)}")
        for line in failure.notes:
            self.note(line)
        self.log(f"FAILED in step: {self.step_name}")
        self.log(f"FAILED: {failure.reason}")

    def ask(self, prompt: str, secret: bool = False) -> str:
        """The line typed at *prompt*, without its newline; with *secret*, typed unseen."""
        if not self.interactive:
            where = "Terminal" if Platform.is_mac() else "a terminal (over ssh: ssh -t)"
            raise InstallFailed("No terminal to ask on.",
                                f"Run the installer from {where}, or set the value in the environment (see the top of install.sh).")
        return self._read(f"  {self.bold(prompt)} ", secret)

    def _read(self, prompt: str, secret: bool) -> str:
        with open(self.TTY, "w") as screen, open(self.TTY) as keyboard:
            if not secret:
                screen.write(prompt)
                screen.flush()
                return keyboard.readline().rstrip("\n")
            fd = keyboard.fileno()
            before = termios.tcgetattr(fd)
            hidden = termios.tcgetattr(fd)
            hidden[3] &= ~termios.ECHO
            try:
                # Echo goes off before the prompt shows: nothing typed at it is ever on the screen.
                termios.tcsetattr(fd, termios.TCSADRAIN, hidden)
                screen.write(prompt)
                screen.flush()
                return keyboard.readline().rstrip("\n")
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, before)
                screen.write("\n")

    @staticmethod
    def today() -> str:
        return datetime.now().strftime("%Y-%m-%d")
