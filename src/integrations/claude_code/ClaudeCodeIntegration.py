import json
import os
import time
from pathlib import Path
from typing import List, Optional

from assistant.integration.Feed import Feed, FeedBatch, FeedEntry
from assistant.integration.Integration import Integration
from assistant.llm import create_llm, parse_llm_config
from common.ConfigManager import config
from common.logging import info, trace, error, fine
from common.Paths import CLAUDE_OWN_SESSIONS_PATH
from .SecretFilter import scrub
from .SessionLog import (Digest, digest, edited_file, entrypoint, git_events, is_scripted, project_label,
                         prompt_text, read_records, record_time, tool_uses, truncate, PROMPT_MAX_CHARS)

DEFAULT_CLAUDE_DIR = "~/.claude"
DEFAULT_IDLE_MINUTES = 10
POLL_INTERVAL = 60
SUMMARY_TIER = "smart"
DIGEST_PROMPTS = 20              # prompts the summary request shows the model, newest last
DIGEST_FILES = 40
FALLBACK_PROMPTS = 3             # prompts a code-built summary names when no model answers
PREFIX = "Claude Code"
SUMMARY_SYSTEM = (
    "You write one entry of a work journal. The user ran a Claude Code session (an AI coding assistant in the "
    "terminal); you get the project, the prompts they typed, the files edited, the git work and the assistant's last "
    "reply. Summarise in two to four plain sentences what they set out to do and what came of it: name the "
    "repository and the main change, mention commits, branches and pull requests when there were any, and say when "
    "something was left unfinished. Past tense, third person (\"{owner} …\"), no headings, no lists, no preamble."
)


class ClaudeCodeIntegration(Integration, Feed):
    """The owner's Claude Code sessions as a journal: prompts, edits, git work.

    Reads the session files Claude Code keeps under ~/.claude and writes what the owner did
    there into Keep's history: the prompts typed, the files edited, the git work, and a few
    sentences of summary once a session goes quiet — so Keep can say what was done there
    today or in a repository last week.

    A Feed (plans/FEED_PLAN.md) over `<claude_dir>/projects/<project>/<session>.jsonl`. The
    cursor is a JSON map of session id → byte offset read, files already announced and the
    offset the last summary covered, so a restart carries on where it stopped and a session
    resumed later gets a second summary of the part after the first. Sessions the server
    starts itself are left out by two marks: `entrypoint` "sdk-cli" in the records (every
    `claude -p` run) and the id list CodeQualityMaintainer writes to CLAUDE_OWN_SESSIONS_PATH.
    A session's end is idleness alone — Claude Code writes no end marker — measured on the
    file's mtime against `claude_code.idle_minutes`. During the catch-up (`backfill`) the
    summary is code-built, not a model call: a month of old sessions is not worth one paid
    call each. Every stored text goes through SecretFilter first."""
    sensitive = True

    poll_interval = POLL_INTERVAL
    item_noun = "events"
    default_backfill_months = 1

    def __init__(self):
        super().__init__()
        self._claude_dir = Path(os.path.expanduser(config.get("claude_code.claude_dir", DEFAULT_CLAUDE_DIR)))
        self._open_error: Optional[str] = None
        self._llm = None

    def _on_reset(self):
        from assistant.integration.FeedRunner import Feeds
        Feeds.get_instance().stop(self.feed_name)

    @property
    def name(self) -> str:
        return "Claude Code"

    @property
    def config_keys(self) -> List[str]:
        return []

    @property
    def projects_dir(self) -> Path:
        return self._claude_dir / "projects"

    def check_health(self) -> bool:
        if not self.projects_dir.is_dir():
            raise RuntimeError(f"{self.projects_dir} is not there — has Claude Code run on this machine? "
                               f"(claude_code.claude_dir points elsewhere when it keeps its sessions elsewhere)")
        trace(f"Claude Code: {self.projects_dir} is readable")
        return True

    # --- Feed ---------------------------------------------------------------------------------

    @property
    def feed_name(self) -> str:
        return "Claude Code"

    @property
    def config_slug(self) -> str:
        return "claude_code"

    @property
    def open_error(self) -> Optional[str]:
        return self._open_error

    def open_feed(self) -> bool:
        try:
            self.check_health()
        except RuntimeError as e:
            self._open_error = str(e)
            error(f"Claude Code: {self._open_error}")
            return False
        self._open_error = None
        return True

    def close_feed(self):
        self._llm = None

    def initial_cursor(self, horizon_ts: float) -> Optional[str]:
        return json.dumps({"horizon": horizon_ts, "sessions": {}})

    @staticmethod
    def idle_seconds() -> float:
        value = config.get("claude_code.idle_minutes", DEFAULT_IDLE_MINUTES)
        try:
            return max(0.0, float(value)) * 60
        except (TypeError, ValueError):
            info(f"Claude Code: claude_code.idle_minutes={value!r} is not a number, using {DEFAULT_IDLE_MINUTES}")
            return DEFAULT_IDLE_MINUTES * 60

    def fetch(self, cursor: Optional[str], backfill: bool) -> FeedBatch:
        """Every session file's new lines since the cursor, as events; then a summary for each
        session that has gone idle since its last one. One pass over all files is one batch:
        the files are read by offset, so a pass is cheap however many sessions there are."""
        state = self._load_state(cursor)
        sessions: dict = state["sessions"]
        own = self._own_sessions()
        now = time.time()
        idle = self.idle_seconds()
        entries: list[FeedEntry] = []
        seen: set[str] = set()
        for session_id, path, size, mtime in self._session_files():
            seen.add(session_id)
            session = sessions.get(session_id)
            if session is None:
                if mtime < state["horizon"]:
                    trace(f"Claude Code: session {session_id[:8]} is older than the horizon, skipped")
                    sessions[session_id] = {"offset": size, "summarized": size, "reported": []}
                    continue
                session = sessions[session_id] = {"offset": 0, "summarized": 0, "reported": []}
            if session.get("scripted"):
                continue
            if size > session["offset"]:
                records, offset = read_records(str(path), session["offset"])
                if offset == session["offset"]:
                    continue
                if not session.get("kind"):
                    kind = entrypoint(records)
                    if kind is None:
                        trace(f"Claude Code: session {session_id[:8]} has no entrypoint yet, waiting")
                        continue
                    if is_scripted(kind) or session_id in own:
                        why = "the server's own" if session_id in own else f"a scripted ({kind}) run"
                        info(f"Claude Code: session {session_id[:8]} is {why}, left out of the journal")
                        session.update(scripted=True, offset=size, summarized=size)
                        continue
                    session["kind"] = kind
                entries.extend(self._events(session_id, session, records, now))
                session["offset"] = offset
            if session["offset"] > session["summarized"] and session["offset"] >= size and now - mtime >= idle:
                entry = self._summary_entry(session_id, path, session, backfill)
                session["summarized"] = session["offset"]
                if entry is not None:
                    entries.append(entry)
        gone = [session_id for session_id in sessions if session_id not in seen]
        for session_id in gone:
            del sessions[session_id]
        if gone:
            trace(f"Claude Code: {len(gone)} session file(s) gone, dropped from the cursor")
        if entries:
            fine(f"Claude Code: {len(entries)} new event(s) from {len(seen)} session file(s)")
        else:
            trace(f"Claude Code: nothing new in {len(seen)} session file(s)")
        return FeedBatch(entries, cursor=json.dumps(state), exhausted=True)

    # --- the pieces ---------------------------------------------------------------------------

    def _load_state(self, cursor: Optional[str]) -> dict:
        try:
            state = json.loads(cursor) if cursor else {}
        except (json.JSONDecodeError, TypeError) as e:
            info(f"Claude Code: the cursor is not JSON ({e}), starting over from the horizon")
            state = {}
        if not isinstance(state, dict):
            state = {}
        state.setdefault("horizon", 0.0)
        if not isinstance(state.get("sessions"), dict):
            state["sessions"] = {}
        return state

    def _session_files(self) -> list[tuple[str, Path, int, float]]:
        """(session id, path, size, mtime) of every session file, oldest change first."""
        found = []
        if not self.projects_dir.is_dir():
            return found
        for path in self.projects_dir.glob("*/*.jsonl"):
            try:
                stat = path.stat()
            except OSError as e:
                trace(f"Claude Code: cannot stat {path.name}: {e}")
                continue
            found.append((path.stem, path, stat.st_size, stat.st_mtime))
        found.sort(key=lambda item: item[3])
        return found

    @staticmethod
    def _own_sessions() -> set[str]:
        try:
            with open(CLAUDE_OWN_SESSIONS_PATH, encoding="utf-8") as handle:
                return {line.strip() for line in handle if line.strip()}
        except FileNotFoundError:
            return set()
        except OSError as e:
            info(f"Claude Code: cannot read {CLAUDE_OWN_SESSIONS_PATH} ({e}), only the entrypoint mark tells the server's runs apart")
            return set()

    @staticmethod
    def _owner() -> str:
        return config.get("owner_name", "The owner")

    def _events(self, session_id: str, session: dict, records: list[dict], now: float) -> list[FeedEntry]:
        """The journal rows for a batch of records: the session's start (at its first prompt),
        each prompt, the git work and the pull requests as they happen, and one row for the
        files edited that the session has not announced before."""
        entries = []
        new_files = []
        last_ts = None
        cwd = session.get("cwd") or ""
        for record in records:
            if not cwd and record.get("cwd"):
                cwd = session["cwd"] = record["cwd"]
            when = record_time(record)
            if when is not None:
                last_ts = when
            ts = when if when is not None else now
            label = f"{PREFIX} [{project_label(cwd)}]"
            text = prompt_text(record)
            if text:
                if not session.get("started"):
                    session["started"] = True
                    branch = record.get("gitBranch")
                    where = f" (branch {branch})" if branch and branch != "HEAD" else ""
                    entries.append(FeedEntry(ts, f"{PREFIX}: {self._owner()} started a session in {project_label(cwd)}{where}"))
                entries.append(FeedEntry(ts, f"{label} {self._owner()} asked: {scrub(truncate(text, PROMPT_MAX_CHARS))}"))
                continue
            for name, tool_input in tool_uses(record):
                path = edited_file(name, tool_input, cwd)
                if path and path not in session["reported"] and path not in new_files:
                    new_files.append(path)
                if name == "Bash":
                    for event in git_events(tool_input.get("command") or ""):
                        entries.append(FeedEntry(ts, f"{label} {scrub(event)}"))
            if record.get("type") == "pr-link" and record.get("prUrl"):
                entries.append(FeedEntry(ts, f"{label} opened pull request {record['prUrl']}"))
        if new_files:
            session["reported"].extend(new_files)
            label = f"{PREFIX} [{project_label(cwd)}]"
            entries.append(FeedEntry(last_ts if last_ts is not None else now,
                                     f"{label} edited {scrub(', '.join(new_files))}"))
        if entries:
            trace(f"Claude Code: session {session_id[:8]}: {len(entries)} event(s), "
                  f"{len(new_files)} file(s) newly edited")
        return entries

    def _summary_entry(self, session_id: str, path: Path, session: dict, backfill: bool) -> Optional[FeedEntry]:
        """The row that closes a session: its span and counts, then the summary — the
        model's while live, a code-built one during the catch-up or when the model fails.
        A session with no prompt, edit or git work in the part since the last summary
        (opened and closed, a `/command` alone) gets no row."""
        try:
            records, _ = read_records(str(path), session["summarized"])
        except OSError as e:
            error(f"Claude Code: cannot read session {session_id[:8]} for its summary: {e}")
            return None
        facts = digest(session_id, records)
        if not facts.prompts and not facts.files and not facts.git:
            trace(f"Claude Code: session {session_id[:8]} went idle with nothing to journal")
            return None
        continued = session["summarized"] > 0
        if backfill:
            trace(f"Claude Code: session {session_id[:8]} summarised without a model call (catch-up)")
            summary = self._fallback_summary(facts)
        else:
            summary = self._model_summary(facts) or self._fallback_summary(facts)
        counts = [f"{len(facts.prompts)} prompt{'s' if len(facts.prompts) != 1 else ''}"]
        if facts.files:
            counts.append(f"{len(facts.files)} file{'s' if len(facts.files) != 1 else ''} edited")
        commits = sum(1 for event in facts.git if event.startswith("committed"))
        if commits:
            counts.append(f"{commits} commit{'s' if commits != 1 else ''}")
        if facts.pr_links:
            counts.append(f"{len(facts.pr_links)} pull request{'s' if len(facts.pr_links) != 1 else ''}")
        span = ", ".join(part for part in (facts.span_text, facts.duration_text) if part)
        head = f"{PREFIX} [{facts.project}] session {'continued and ' if continued else ''}ended"
        content = f"{head} ({span}; {', '.join(counts)}): {scrub(summary)}"
        ended = facts.ended if facts.ended is not None else time.time()
        info(f"Claude Code: session {session_id[:8]} in {facts.project} ended, summary written ({len(content)} chars)")
        return FeedEntry(ended + 1, content)

    def _model_summary(self, facts: Digest) -> Optional[str]:
        try:
            llm = self._summary_llm()
            if llm is None:
                return None
            reply = llm.think(system_message=SUMMARY_SYSTEM.format(owner=self._owner()),
                              user_message=self._render_digest(facts))
            reply = " ".join((reply or "").split())
            if not reply:
                info(f"Claude Code: the model answered nothing for session {facts.session_id[:8]}, using the digest")
                return None
            return reply
        except Exception as e:
            error(f"Claude Code: summary of session {facts.session_id[:8]} failed ({type(e).__name__}: {e}), using the digest")
            return None

    def _summary_llm(self):
        if self._llm is None:
            name, model = parse_llm_config(SUMMARY_TIER)
            llm = create_llm(name, model, prompt_cache=False)
            missing = llm.missing_api_key()
            if missing:
                info(f"Claude Code: no model for the summaries ({missing} is not set), using the digest")
                return None
            llm.initialize()
            self._llm = llm
            trace(f"Claude Code: summaries on {name}{f' ({model})' if model else ''}")
        return self._llm

    def _render_digest(self, facts: Digest) -> str:
        """The facts of the session as the model sees them, credentials blanked."""
        lines = [f"Repository / folder: {facts.project}" + (f" (branch {facts.branch})" if facts.branch else "")]
        if facts.title:
            lines.append(f"Title Claude Code gave the session: {facts.title}")
        if facts.span_text:
            lines.append(f"When: {facts.span_text} ({facts.duration_text})")
        prompts = facts.prompts[-DIGEST_PROMPTS:]
        skipped = len(facts.prompts) - len(prompts)
        lines.append(f"Prompts typed ({len(facts.prompts)}{f', the first {skipped} left out' if skipped else ''}):")
        lines.extend(f"- {prompt}" for prompt in prompts)
        if facts.files:
            shown = facts.files[:DIGEST_FILES]
            more = f" and {len(facts.files) - len(shown)} more" if len(facts.files) > len(shown) else ""
            lines.append(f"Files edited: {', '.join(shown)}{more}")
        if facts.lines_added is not None or facts.lines_removed is not None:
            lines.append(f"Lines added / removed: {facts.lines_added or 0} / {facts.lines_removed or 0}")
        if facts.git:
            lines.append("Git: " + "; ".join(facts.git))
        if facts.pr_links:
            lines.append("Pull requests: " + ", ".join(facts.pr_links))
        if facts.last_reply:
            lines.append(f"The assistant's last reply: {facts.last_reply}")
        return scrub("\n".join(lines))

    @staticmethod
    def _fallback_summary(facts: Digest) -> str:
        """A summary built from the facts alone: the title or the first prompts, the files, the git work."""
        parts = []
        if facts.title:
            parts.append(facts.title)
        prompts = facts.prompts[:FALLBACK_PROMPTS]
        if prompts:
            more = f" (+{len(facts.prompts) - len(prompts)} more)" if len(facts.prompts) > len(prompts) else ""
            parts.append("asked: " + " | ".join(truncate(prompt, 160) for prompt in prompts) + more)
        if facts.files:
            shown = facts.files[:8]
            more = f" and {len(facts.files) - len(shown)} more" if len(facts.files) > len(shown) else ""
            parts.append("edited " + ", ".join(shown) + more)
        if facts.git:
            parts.append("; ".join(facts.git))
        return ". ".join(parts) + "."
