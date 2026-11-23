import html
import json
import os
import re
import time
from datetime import datetime

from common import Version
from common.logging import info, error, trace
from common.Paths import CACHE_DIR

STATE_FILE = CACHE_DIR / "last_startup.json"
# Written by a restart the assistant started from a chat, taken by the next boot: the
# conversation it left hanging is resumed with a system turn (issue #113).
RESUME_FILE = CACHE_DIR / "resume_after_restart.json"
_MARKUP = re.compile(r"</?[bi]>")   # the only tags the release note's Telegram HTML dialect uses


class StartupTracker:
    """Persists the server's last startup time and builds the startup/shutdown history messages.

    The startup timestamp lives in cache/last_startup.json; on boot the commits made
    since then are pulled from git so the assistant can see what changed. The same file
    remembers which release's what's new (the note release.sh wrote into the tag) has been
    announced, so the first start on a release tells the household and a later restart does not.
    """

    @staticmethod
    def _read_state() -> dict:
        if not STATE_FILE.exists():
            info("StartupTracker: no previous startup record found, this looks like a first boot.")
            return {}
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError, json.JSONDecodeError) as e:
            error(f"StartupTracker: failed to read state file ({e}); treating as first boot.")
            return {}

    @staticmethod
    def read_previous_startup() -> float | None:
        try:
            ts = float(StartupTracker._read_state().get("started_at"))
        except (ValueError, TypeError):
            return None
        trace(f"StartupTracker: previous startup was at {datetime.fromtimestamp(ts).isoformat(timespec='seconds')}.")
        return ts

    @staticmethod
    def read_announced_release() -> str | None:
        """The version whose what's new an earlier start announced, None when none has been."""
        announced = StartupTracker._read_state().get("announced_release")
        return announced if isinstance(announced, str) and announced else None

    @staticmethod
    def write_current_startup(ts: float, announced_release: str | None = None):
        """Record this start; *announced_release* names the release whose what's new this
        start announced (or skipped, on a first boot), None keeps whatever an earlier start recorded."""
        if announced_release is None:
            announced_release = StartupTracker.read_announced_release()
        state = {"started_at": ts}
        if announced_release:
            state["announced_release"] = announced_release
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = STATE_FILE.with_suffix(".json.tmp")
            with open(tmp_path, "w") as f:
                json.dump(state, f)
            os.replace(tmp_path, STATE_FILE)
            trace(f"StartupTracker: recorded current startup time at {STATE_FILE}.")
        except OSError as e:
            error(f"StartupTracker: failed to write state file: {e}")

    @staticmethod
    def unannounced_release_notes() -> str:
        """The what's new of the running release when this is the first start on it; "" when the
        commit has no release note, an earlier start announced it already, or this is the very
        first boot — a fresh install has nothing to compare the release to, so the household's
        first message is a greeting, not a change log."""
        notes = Version.release_notes()
        if not notes:
            trace("StartupTracker: the running commit carries no what's new.")
            return ""
        if StartupTracker.read_previous_startup() is None:
            info(f"StartupTracker: first boot ever on {Version.version()}, its what's new stays unsent.")
            return ""
        if StartupTracker.read_announced_release() == Version.version():
            info(f"StartupTracker: the what's new of {Version.version()} was announced by an earlier start.")
            return ""
        info(f"StartupTracker: first start on {Version.version()}, its what's new goes to the household.")
        return notes

    @staticmethod
    def write_resume_marker(reason: str):
        """Called by `CodeQualityMaintainer.restart` right before it spawns start.sh for a restart
        the assistant started from a chat, so only an actual restart leaves the marker."""
        try:
            RESUME_FILE.parent.mkdir(parents=True, exist_ok=True)
            RESUME_FILE.write_text(json.dumps({"reason": reason, "at": time.time()}))
            info(f"StartupTracker: resume marker written ({reason}) — the next boot resumes the conversation.")
        except OSError as e:
            error(f"StartupTracker: failed to write the resume marker: {e}")

    @staticmethod
    def resume_pending() -> bool:
        """Whether a restart left a marker `take_resume_marker` has not taken yet."""
        return RESUME_FILE.exists()

    @staticmethod
    def take_resume_marker() -> dict | None:
        """The marker the restart left, taken once: None when there is none (a nightly update, a
        crash, a manual start) or it cannot be read."""
        try:
            data = json.loads(RESUME_FILE.read_text())
        except FileNotFoundError:
            trace("StartupTracker: no resume marker, no conversation to resume.")
            return None
        except (OSError, ValueError) as e:
            error(f"StartupTracker: unreadable resume marker ({e}), dropped.")
            data = None
        try:
            RESUME_FILE.unlink()
        except OSError as e:
            error(f"StartupTracker: failed to remove the resume marker: {e}")
        if not isinstance(data, dict) or not isinstance(data.get("at"), (int, float)):
            return None
        info(f"StartupTracker: resume marker found ({data.get('reason')!r}), the conversation is resumed.")
        return data

    @staticmethod
    def build_resume_turn(marker: dict, version: str, now: float | None = None) -> str:
        """The system turn that resumes the conversation a restart interrupted. The household
        already has the startup announcement, in the history too, so the turn carries only
        the restart itself and the ask: report the outcome, keep the promises."""
        now = time.time() if now is None else now
        at = time.strftime("%H:%M", time.localtime(marker["at"]))
        down = max(0, int(round(now - marker["at"])))
        reason = marker.get("reason") or "no recorded reason"
        return (f"You restarted the server at {at} ({reason}). It is up again on {version} after "
                f"{down} seconds. The household was already sent the startup announcement, which is in "
                f"the history above. Continue the conversation you were in: report the outcome of the "
                f"restart and do what you said you would do after it.")

    @staticmethod
    def plain(notes: str) -> str:
        """The note without its chat markup, for the history row the LLM reads."""
        return html.unescape(_MARKUP.sub("", notes))

    @staticmethod
    def get_commits_since(ts: float) -> list[str]:
        from assistant.agents.CodeQualityMaintainer import CodeQualityMaintainer, GIT_TIMEOUT
        try:
            result = CodeQualityMaintainer._run_subprocess(
                ["git", "log", f"--since=@{int(ts)}", "--pretty=format:%h %s"],
                "startup-log",
                GIT_TIMEOUT,
            )
        except Exception as e:
            error(f"StartupTracker: git log failed: {e}")
            return []

        if result.returncode != 0:
            error(f"StartupTracker: git log returned {result.returncode}: {result.stderr.strip() if result.stderr else ''}")
            return []

        output = (result.stdout or "").strip()
        if not output:
            trace("StartupTracker: no commits since last start.")
            return []

        commits = output.splitlines()
        info(f"StartupTracker: {len(commits)} commit(s) since last start.")
        return commits

    @staticmethod
    def record_shutdown():
        """Persist a shutdown marker directly to the history DB.

        Writes via FloxyDatabase rather than the History prompt section so this is safe
        to call from atexit, after section monitoring threads may have stopped.
        """
        try:
            from .FloxyDatabase import FloxyDatabase
            FloxyDatabase.get_instance().add_history_entry(
                "system", "Server is shutting down.", time.time(), "shutdown"
            )
            info("StartupTracker: recorded shutdown marker in history.")
        except Exception as e:
            error(f"StartupTracker: failed to record shutdown marker: {e}")

    @staticmethod
    def build_history_message(commits: list[str], previous_ts: float | None, whats_new: str = "") -> str:
        """The startup row the LLM reads. *whats_new* is the release note the household is
        being sent on this start, so the assistant knows what they were told and need not repeat it."""
        if previous_ts is None:
            message = "Server started up successfully."
        elif not commits:
            message = "Server started up successfully. No changes since last start."
        else:
            change_log = "\n".join(commits)
            message = f"Server started up successfully. Change log since last start:\n{change_log}"
        if whats_new:
            message += (f"\n\nThis is the first start on release {Version.version()}; the household was sent "
                        f"this what's new:\n{StartupTracker.plain(whats_new)}")
        return message

    @staticmethod
    def build_announcement(commits: list[str], version: str, whats_new: str = "", trial_note: str = "") -> str:
        """What the owner's chats are told once the server is up: that it is running again,
        on which version, and the subjects of the commits picked up since the last start — or,
        on the first start of a release, its what's new in their place (already in the
        Notifier's HTML dialect; the subjects are escaped into it). *trial_note* is what a
        PR trial adds (`PullRequestTrial.announcement_note`), a paragraph of its own at the end."""
        up = f"Server is up again — {html.escape(version)}."
        if whats_new:
            text = f"{up} What's new in this release:\n\n{whats_new}"
        elif not commits:
            text = f"{up} No new commits since the last start."
        else:
            subjects = "\n".join(f"• {html.escape(c.partition(' ')[2] or c, quote=False)}" for c in commits)
            count = f"{len(commits)} new commit{'s' if len(commits) != 1 else ''}"
            text = f"{up} {count} since the last start:\n{subjects}"
        return f"{text}\n\n{trial_note}" if trial_note else text
