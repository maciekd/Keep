"""The installer's lines, questions, progress and log, as `install.sh` draws them."""
import os
import re
import shutil
import subprocess
import sys
import termios
import threading
import time
from collections import deque
from datetime import datetime

from common import Palette, Platform

ACCENTS = {"green": (Palette.GREEN, "32"), "red": (Palette.RED, "31"), "yellow": (Palette.YELLOW, "33"),
           "cyan": (Palette.CYAN, "36")}      # each with the plain ANSI colour a terminal of 16 falls back to
TRUE_COLOUR, CUBE, BASIC = "true", "256", "basic"
ANSI = re.compile(r"\033\[[0-9;]*m")      # the colours are the screen's, not the log's
FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
TICK = 0.1
GB, MB = 10 ** 9, 10 ** 6      # as a download is counted: Hugging Face's sizes and a disk's


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
        self.clock = time.monotonic
        self.live = self.out.isatty() and not env.get("NO_COLOR") and env.get("TERM", "dumb") != "dumb"
        self.depth = self._depth(env)
        codes = {"bold": "1", "dim": "2", "reset": "0", **{name: self._sgr(rgb, basic) for name, (rgb, basic) in ACCENTS.items()}}
        self._codes = {name: f"\033[{code}m" if self.live else "" for name, code in codes.items()}
        self.interactive = self._has_terminal()

    @staticmethod
    def _depth(env) -> str:
        """How many colours the terminal has: all of them where it says so (`COLORTERM`), the cube
        of 256 where its name does — Terminal before macOS 26 — and the 16 otherwise."""
        if env.get("COLORTERM", "") in ("truecolor", "24bit"):
            return TRUE_COLOUR
        return CUBE if "256color" in env.get("TERM", "") else BASIC

    def _sgr(self, rgb: tuple, basic: str) -> str:
        if self.depth == TRUE_COLOUR:
            return "38;2;{};{};{}".format(*rgb)
        return f"38;5;{Palette.nearest_256(rgb)}" if self.depth == CUBE else basic

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
            f.write(ANSI.sub("", text) + "\n")

    def report(self, name: str, value) -> None:
        """A fact for the install report: `install.sh` reads these lines of this run back from the
        log and sends them with its own event (it holds the telemetry, and the opt-out)."""
        self.log(f"report: {name}={' '.join(str(value).split())}")

    def step(self, name: str) -> None:
        self.step_number += 1
        self.step_name = name
        self._print(f"\n  {self.styled('dim', f'[{self.step_number}/{self.steps}]')} {self.bold(name)}")
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

    def columns(self) -> int:
        return shutil.get_terminal_size((80, 24)).columns

    def _redraw(self, line: str) -> None:
        """*line* in place of the one drawn before; it has to fit the row, or every redraw leaves
        the last one behind."""
        self.out.write(f"\r{line}\033[K")
        self.out.flush()

    def _cursor(self, shown: bool) -> None:
        self.out.write("\r\033[K\033[?25h" if shown else "\033[?25l")
        self.out.flush()

    def wait(self, label: str, work):
        """Runs *work* while the line shows a spinner, *label* and the time it has taken — for what
        has no total to draw a bar of. Returns what *work* returned and raises what it raised."""
        outcome = {}

        def body():
            try:
                outcome["result"] = work()
            except BaseException as e:      # the caller's to see, whatever it is
                outcome["error"] = e

        thread = threading.Thread(target=body, daemon=True, name="install-wait")
        started = self.clock()
        thread.start()
        if not self.live:
            self._print(f"  … {label}")
            thread.join()
        else:
            self._cursor(False)
            try:
                frame = 0
                while thread.is_alive():
                    elapsed = f"({fmt_elapsed(self.clock() - started)})"
                    text = clip(label, self.columns() - len(elapsed) - 6)
                    self._redraw(f"  {self.styled('cyan', FRAMES[frame % len(FRAMES)])} {text} {self.styled('dim', elapsed)}")
                    frame += 1
                    thread.join(TICK)
            finally:
                self._cursor(True)
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("result")

    def run(self, label: str, argv: list, env: dict | None = None) -> bool:
        """Whether the command succeeded. Its output goes to the log and the screen shows the
        spinner; after a failure, the log's last lines."""
        self.log("")
        self.log(f"--- {label}: {' '.join(argv)}")
        started = self.clock()

        def command() -> int:
            with open(self.log_path or os.devnull, "a") as log:
                try:
                    return subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env).returncode
                except OSError as e:
                    log.write(f"{type(e).__name__}: {e}\n")
                    return 127

        status = self.wait(label, command)
        if status == 0:
            self.ok(f"{label} {self.styled('dim', '(' + fmt_elapsed(self.clock() - started) + ')')}")
            return True
        self._print(f"\n  {self.styled('red', f'✖ {label} failed (exit {status}). Last lines of the log:')}\n")
        for line in self._log_tail(25):
            self._print(f"    │ {line}")
        self.log(f"{label}: exit {status}")
        return False

    def _log_tail(self, count: int) -> list[str]:
        if not self.log_path:
            return []
        with open(self.log_path, errors="replace") as f:
            return f.read().splitlines()[-count:]

    def bar(self, label: str, total: int) -> "Bar":
        return Bar(self, label, total)


class Bar:
    """The line of a download whose size is known: how far, how fast and how long still. `update`
    takes the bytes there so far — a download that resumes starts where it was —, `close` clears
    the line for what the caller says of the result."""

    WIDTH = 30
    NARROW = 12              # the bar gives way before a fact does: 80 columns hold them all
    WINDOW_SECONDS = 15      # the rate is that of the last quarter minute, not of the whole download
    STEP = 10                # without a terminal to redraw on: a line at each tenth

    def __init__(self, terminal: Terminal, label: str, total: int):
        self.terminal = terminal
        self.label = label
        self.total = max(1, total)
        self.samples = deque()
        self.said = -1
        if terminal.live:
            terminal._cursor(False)

    def update(self, done: int) -> None:
        t = self.terminal
        done = max(0, min(done, self.total))
        now = t.clock()
        self.samples.append((now, done))
        while len(self.samples) > 2 and now - self.samples[0][0] > self.WINDOW_SECONDS:
            self.samples.popleft()
        percent = done * 100 // self.total
        if t.live:
            t._redraw(self.line(done, percent))
        elif percent // self.STEP > self.said:
            self.said = percent // self.STEP
            t._print(f"  … {self.label} {percent}%")

    def rate(self) -> float:
        """Bytes a second over the window; 0 until two readings stand apart."""
        (then, before), (now, done) = self.samples[0], self.samples[-1]
        return (done - before) / (now - then) if now > then else 0.0

    def facts(self, done: int) -> list[str]:
        """What stands after the percentage, the first the one kept longest on a narrow terminal."""
        facts = [f"{done / GB:.1f} of {self.total / GB:.1f} GB"]
        rate = self.rate()
        if rate > 0:
            facts.append(f"{rate / MB:.0f} MB/s")
            if done < self.total:
                facts.append(fmt_left((self.total - done) / rate))
        return facts

    def line(self, done: int, percent: int) -> str:
        t = self.terminal
        facts = self.facts(done)
        room = t.columns() - 1
        fixed = 2 + len(self.label) + 3 + 2 + 4      # the margin, the label, its gap, the bar's gap, "100%"
        while facts and fixed + self.NARROW + sum(3 + len(fact) for fact in facts) > room:
            facts.pop()
        width = max(self.NARROW, min(self.WIDTH, room - fixed - sum(3 + len(fact) for fact in facts)))
        filled = done * width // self.total
        return (f"  {self.label}   {self.cells(filled, width)}  {percent:3d}%"
                + "".join(f"   {fact}" for fact in facts))

    def cells(self, filled: int, width: int) -> str:
        """The bar: where the terminal has every colour it fades up from the accent's deep
        companion, as the console's meters do."""
        t = self.terminal
        rest = t.styled("dim", "░" * (width - filled))
        if not t.live or t.depth != TRUE_COLOUR:
            return t.styled("cyan", "█" * filled) + rest
        deep = Palette.DEEP[Palette.CYAN]
        drawn = "".join("\033[38;2;{};{};{}m█".format(*Palette.mix(deep, Palette.CYAN, i / max(1, width - 1)))
                        for i in range(filled))
        return f"{drawn}\033[0m{rest}"

    def close(self) -> None:
        if self.terminal.live:
            self.terminal._cursor(True)


def clip(text: str, room: int) -> str:
    return text if len(text) <= room or room < 4 else text[:room - 1] + "…"


def fmt_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def fmt_left(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes < 1:
        return "under a minute left"
    if minutes < 60:
        return f"{minutes} min left"
    return f"{minutes // 60} h {minutes % 60:02d} min left"
