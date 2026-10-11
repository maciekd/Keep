"""The Claude Code feed over a fictional ~/.claude: the owner Marcus's sessions in ~/Keep and
~/garden, a scripted run of the server's own, and the session files growing between polls."""
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from integrations.claude_code import ClaudeCodeIntegration as module
from integrations.claude_code.ClaudeCodeIntegration import ClaudeCodeIntegration
from integrations.claude_code.SessionLog import digest, git_events, prompt_text, read_records, record_time

HOME = "/Users/marcus"
T0 = 1_790_000_000.0


def iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) + ".000Z"


def user(text, ts, cwd=f"{HOME}/Keep", entrypoint="cli", branch="feat/lamps", **extra):
    return {"type": "user", "cwd": cwd, "gitBranch": branch, "entrypoint": entrypoint, "userType": "external",
            "isSidechain": False, "timestamp": iso(ts), "sessionId": "s", "uuid": "u",
            "message": {"role": "user", "content": text}, **extra}


def tool_result(ts, cwd=f"{HOME}/Keep"):
    return {"type": "user", "cwd": cwd, "entrypoint": "cli", "isSidechain": False, "timestamp": iso(ts),
            "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}}


def assistant(blocks, ts, cwd=f"{HOME}/Keep"):
    return {"type": "assistant", "cwd": cwd, "entrypoint": "cli", "isSidechain": False, "timestamp": iso(ts),
            "message": {"role": "assistant", "content": blocks}}


def edit(path, ts):
    return assistant([{"type": "tool_use", "name": "Edit", "input": {"file_path": path, "old_string": "a", "new_string": "b"}}], ts)


def bash(command, ts):
    return assistant([{"type": "tool_use", "name": "Bash", "input": {"command": command}}], ts)


def lines(*records) -> str:
    return "".join(json.dumps(record) + "\n" for record in records)


@pytest.fixture(autouse=True)
def reset_singleton():
    ClaudeCodeIntegration.reset_instance()
    yield
    ClaudeCodeIntegration.reset_instance()


@pytest.fixture
def claude_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", HOME, 1) if p.startswith("~") else p)
    root = tmp_path / "claude"
    (root / "projects").mkdir(parents=True)
    monkeypatch.setattr(module, "CLAUDE_OWN_SESSIONS_PATH", tmp_path / "own.txt")
    return root


@pytest.fixture
def feed(claude_dir):
    values = {"claude_code.claude_dir": str(claude_dir), "claude_code.idle_minutes": 10, "owner_name": "Marcus"}
    with patch.object(module.config, "get", side_effect=lambda key, default=None: values.get(key, default)):
        integration = ClaudeCodeIntegration()
        yield integration


def session_file(claude_dir: Path, project: str, session_id: str, text: str, mtime: float) -> Path:
    folder = claude_dir / "projects" / project
    folder.mkdir(exist_ok=True)
    path = folder / f"{session_id}.jsonl"
    path.write_text(text)
    os.utime(path, (mtime, mtime))
    return path


def fetch(feed, cursor=None, now=T0 + 3600, backfill=False):
    with patch.object(module.time, "time", return_value=now):
        return feed.fetch(cursor or feed.initial_cursor(T0 - 86400), backfill=backfill)


@pytest.mark.unit
class TestSessionLog:
    def test_reads_only_complete_lines_and_returns_the_next_offset(self, tmp_path):
        path = tmp_path / "s.jsonl"
        path.write_text(lines(user("one", T0)) + '{"type": "user", "partial')
        records, offset = read_records(str(path), 0)
        assert [r["message"]["content"] for r in records] == ["one"]
        assert offset == len(lines(user("one", T0)).encode())
        path.write_text(path.read_text() + '"}\n' + lines(user("two", T0 + 1)))
        records, offset = read_records(str(path), offset)
        assert [r.get("message", {}).get("content") for r in records][-1] == "two"
        assert offset == path.stat().st_size

    def test_a_tool_result_a_sidechain_and_a_slash_command_are_not_prompts(self):
        assert prompt_text(tool_result(T0)) is None
        assert prompt_text(user("sub task", T0, isSidechain=True)) is None
        assert prompt_text(user("<command-name>/commit</command-name>", T0)) is None
        assert prompt_text(user([{"type": "text", "text": "fix the lamp"}], T0)) == "fix the lamp"

    def test_record_time_reads_iso_and_millisecond_stamps(self):
        assert record_time({"timestamp": "2026-10-11T16:05:25.941Z"}) == pytest.approx(1791734725.941)
        assert record_time({"timestamp": 1791734725941}) == pytest.approx(1791734725.941)
        assert record_time({}) is None

    @pytest.mark.parametrize("command, expected", [
        ('git commit -m "feat: lamps dim"', ['committed "feat: lamps dim"']),
        ("git add -A && git commit -m \"$(cat <<'EOF'\nfix: the hall lamp\n\nbody\nEOF\n)\"", ['committed "fix: the hall lamp"']),
        ("git checkout -b feat/lamps && git push -u origin feat/lamps", ["created branch feat/lamps", "pushed feat/lamps"]),
        ("git switch -c fix/x", ["created branch fix/x"]),
        ("gh pr create --title x", ["opened a pull request"]),
        ("git status && ls", []),
    ])
    def test_git_events(self, command, expected):
        assert git_events(command) == expected

    def test_digest_collects_prompts_files_git_and_the_span(self, claude_dir):
        records = [user("make the lamps dim", T0), edit(f"{HOME}/Keep/src/a.py", T0 + 10),
                   edit(f"{HOME}/Keep/src/a.py", T0 + 20), edit("/etc/hosts", T0 + 30),
                   bash('git commit -m "feat: dim"', T0 + 40), {"type": "ai-title", "aiTitle": "Dimming lamps"},
                   {"type": "pr-link", "prUrl": "https://github.example/pr/7", "timestamp": iso(T0 + 50)},
                   assistant([{"type": "text", "text": "Done, the lamps dim."}], T0 + 60)]
        facts = digest("abc", records)
        assert facts.project == "~/Keep" and facts.branch == "feat/lamps" and facts.title == "Dimming lamps"
        assert facts.prompts == ["make the lamps dim"]
        assert facts.files == ["src/a.py", "/etc/hosts"]
        assert facts.git == ['committed "feat: dim"'] and facts.pr_links == ["https://github.example/pr/7"]
        assert facts.started == T0 and facts.ended == T0 + 60 and facts.duration_text == "1 min"
        assert facts.last_reply == "Done, the lamps dim."


@pytest.mark.unit
class TestFetch:
    def test_an_interactive_session_becomes_start_prompt_edit_and_git_rows(self, feed, claude_dir):
        session_file(claude_dir, "-Users-marcus-Keep", "a1", lines(
            user("make the lamps dim", T0), edit(f"{HOME}/Keep/src/a.py", T0 + 10), tool_result(T0 + 11),
            edit(f"{HOME}/Keep/src/b.py", T0 + 20), bash('git checkout -b feat/lamps', T0 + 30),
            user("now commit it", T0 + 40), bash('git commit -m "feat: lamps dim"', T0 + 50)), T0 + 50)
        batch = fetch(feed, now=T0 + 60)
        contents = [e.content for e in batch.entries]
        assert contents[0] == "Claude Code: Marcus started a session in ~/Keep (branch feat/lamps)"
        assert "Claude Code [~/Keep] Marcus asked: make the lamps dim" in contents
        assert "Claude Code [~/Keep] created branch feat/lamps" in contents
        assert 'Claude Code [~/Keep] committed "feat: lamps dim"' in contents
        assert "Claude Code [~/Keep] edited src/a.py, src/b.py" in contents
        assert batch.exhausted and not any("ended" in c for c in contents)
        assert [e.timestamp for e in batch.entries][:2] == [T0, T0]

    def test_a_scripted_run_and_the_servers_own_session_are_left_out(self, feed, claude_dir, tmp_path):
        session_file(claude_dir, "-Users-marcus-Keep", "sdk1", lines(
            user("nightly analysis", T0, entrypoint="sdk-cli"), edit(f"{HOME}/Keep/x.py", T0 + 1)), T0 + 1)
        session_file(claude_dir, "-Users-marcus-Keep", "own1", lines(
            user("self improvement", T0, entrypoint="cli"), edit(f"{HOME}/Keep/y.py", T0 + 1)), T0 + 1)
        session_file(claude_dir, "-Users-marcus-garden", "mine", lines(
            user("plant tomatoes", T0 + 5, cwd=f"{HOME}/garden", branch="HEAD")), T0 + 5)
        (tmp_path / "own.txt").write_text("own1\n")
        batch = fetch(feed, now=T0 + 10)
        contents = [e.content for e in batch.entries]
        assert contents == ["Claude Code: Marcus started a session in ~/garden",
                            "Claude Code [~/garden] Marcus asked: plant tomatoes"]
        state = json.loads(batch.cursor)["sessions"]
        assert state["sdk1"]["scripted"] and state["own1"]["scripted"] and "scripted" not in state["mine"]

    def test_the_cursor_carries_the_offset_so_a_second_poll_sees_only_the_new_lines(self, feed, claude_dir):
        path = session_file(claude_dir, "-Users-marcus-Keep", "a1", lines(
            user("first", T0), edit(f"{HOME}/Keep/a.py", T0 + 1)), T0 + 1)
        first = fetch(feed, now=T0 + 5)
        assert len(first.entries) == 3
        again = fetch(feed, cursor=first.cursor, now=T0 + 6)
        assert again.entries == [] and json.loads(again.cursor) == json.loads(first.cursor)
        path.write_text(path.read_text() + lines(user("second", T0 + 10), edit(f"{HOME}/Keep/a.py", T0 + 11),
                                                 edit(f"{HOME}/Keep/b.py", T0 + 12)))
        os.utime(path, (T0 + 12, T0 + 12))
        third = fetch(feed, cursor=first.cursor, now=T0 + 15)
        assert [e.content for e in third.entries] == ["Claude Code [~/Keep] Marcus asked: second",
                                                      "Claude Code [~/Keep] edited b.py"]
        assert json.loads(third.cursor)["sessions"]["a1"]["offset"] == path.stat().st_size

    def test_a_session_older_than_the_horizon_is_skipped_but_its_later_lines_are_read(self, feed, claude_dir):
        path = session_file(claude_dir, "-Users-marcus-Keep", "old", lines(user("ancient", T0 - 10 * 86400)), T0 - 10 * 86400)
        batch = fetch(feed, now=T0)
        assert batch.entries == []
        path.write_text(path.read_text() + lines(user("back again", T0)))
        os.utime(path, (T0, T0))
        later = fetch(feed, cursor=batch.cursor, now=T0 + 1)
        assert [e.content for e in later.entries][-1] == "Claude Code [~/Keep] Marcus asked: back again"

    def test_a_file_that_disappears_leaves_the_cursor(self, feed, claude_dir):
        path = session_file(claude_dir, "-Users-marcus-Keep", "gone", lines(user("hi", T0)), T0)
        batch = fetch(feed, now=T0 + 1)
        assert "gone" in json.loads(batch.cursor)["sessions"]
        path.unlink()
        assert "gone" not in json.loads(fetch(feed, cursor=batch.cursor, now=T0 + 2).cursor)["sessions"]


@pytest.mark.unit
class TestSummary:
    def _session(self, claude_dir):
        return session_file(claude_dir, "-Users-marcus-Keep", "a1", lines(
            user("make the lamps dim", T0), edit(f"{HOME}/Keep/src/a.py", T0 + 60),
            bash('git commit -m "feat: dim"', T0 + 120), {"type": "custom-title", "customTitle": "Dim lamps"},
            assistant([{"type": "text", "text": "The lamps now dim; committed."}], T0 + 180)), T0 + 180)

    def test_no_summary_until_the_session_has_been_idle_long_enough(self, feed, claude_dir):
        self._session(claude_dir)
        with patch.object(feed, "_model_summary", return_value="dimmed") as model:
            early = fetch(feed, now=T0 + 180 + 9 * 60)
            assert not any("ended" in e.content for e in early.entries)
            late = fetch(feed, cursor=early.cursor, now=T0 + 180 + 11 * 60)
        model.assert_called_once()
        assert len(late.entries) == 1 and json.loads(late.cursor)["sessions"]["a1"]["summarized"] > 0
        assert fetch(feed, cursor=late.cursor, now=T0 + 180 + 30 * 60).entries == []

    def test_the_model_summary_closes_the_session_row(self, feed, claude_dir):
        self._session(claude_dir)
        with patch.object(feed, "_summary_llm") as llm:
            llm.return_value.think.return_value = "  Marcus made the lamps dim and committed it.\n"
            batch = fetch(feed, now=T0 + 180 + 11 * 60)
        content = batch.entries[-1].content
        assert content.startswith("Claude Code [~/Keep] session ended (")
        assert "1 prompt, 1 file edited, 1 commit" in content
        assert content.endswith(": Marcus made the lamps dim and committed it.")
        assert batch.entries[-1].timestamp == T0 + 181
        sent = llm.return_value.think.call_args.kwargs
        assert "Marcus" in sent["system_message"] and "src/a.py" in sent["user_message"] and "Dim lamps" in sent["user_message"]

    def test_the_catch_up_builds_the_summary_without_a_model_call(self, feed, claude_dir):
        self._session(claude_dir)
        with patch.object(feed, "_summary_llm") as llm:
            batch = fetch(feed, now=T0 + 180 + 11 * 60, backfill=True)
        llm.assert_not_called()
        content = batch.entries[-1].content
        assert "Dim lamps" in content and "make the lamps dim" in content and 'committed "feat: dim"' in content

    def test_a_failing_model_falls_back_to_the_digest(self, feed, claude_dir):
        self._session(claude_dir)
        with patch.object(feed, "_summary_llm") as llm:
            llm.return_value.think.side_effect = RuntimeError("boom")
            batch = fetch(feed, now=T0 + 180 + 11 * 60)
        assert "src/a.py" in batch.entries[-1].content

    def test_a_resumed_session_gets_a_second_summary_of_the_new_part(self, feed, claude_dir):
        path = self._session(claude_dir)
        with patch.object(feed, "_model_summary", return_value="first part"):
            first = fetch(feed, now=T0 + 180 + 11 * 60)
        path.write_text(path.read_text() + lines(user("also brighten them", T0 + 7200), edit(f"{HOME}/Keep/src/c.py", T0 + 7210)))
        os.utime(path, (T0 + 7210, T0 + 7210))
        with patch.object(feed, "_model_summary", return_value="second part") as model:
            second = fetch(feed, cursor=first.cursor, now=T0 + 7210 + 11 * 60)
        facts = model.call_args.args[0]
        assert facts.prompts == ["also brighten them"] and facts.files == ["src/c.py"]
        assert second.entries[-1].content.startswith("Claude Code [~/Keep] session continued and ended")

    def test_an_idle_session_with_nothing_typed_gets_no_row(self, feed, claude_dir):
        session_file(claude_dir, "-Users-marcus-Keep", "empty", lines(
            user("<command-name>/help</command-name>", T0), {"type": "mode", "mode": "normal"}), T0)
        with patch.object(feed, "_model_summary") as model:
            batch = fetch(feed, now=T0 + 20 * 60)
        model.assert_not_called()
        assert batch.entries == []


@pytest.mark.unit
class TestSecretsInRows:
    def test_a_key_in_a_prompt_is_blanked_before_the_row(self, feed, claude_dir):
        session_file(claude_dir, "-Users-marcus-Keep", "k", lines(
            user("use the key sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789 for the call", T0)), T0)
        batch = fetch(feed, now=T0 + 1)
        prompt = [e.content for e in batch.entries if "asked" in e.content][0]
        assert "sk-ant" not in prompt and "[secret]" in prompt


@pytest.mark.unit
class TestIntegration:
    def test_health_needs_the_projects_folder(self, feed, claude_dir):
        assert feed.check_health() is True
        (claude_dir / "projects").rmdir()
        with pytest.raises(RuntimeError, match="projects"):
            feed.check_health()
        assert feed.open_feed() is False and "projects" in feed.open_error

    def test_idle_minutes_falls_back_on_nonsense(self, feed):
        with patch.object(module.config, "get", side_effect=lambda key, default=None: "soon"):
            assert feed.idle_seconds() == module.DEFAULT_IDLE_MINUTES * 60
        with patch.object(module.config, "get", side_effect=lambda key, default=None: 2.5):
            assert feed.idle_seconds() == 150

    def test_example_config_documents_the_section(self):
        from common import ConfigSchema
        assert "claude_code" in ConfigSchema.sections()
        assert ConfigSchema.leaves("claude_code") == ["claude_code.idle_minutes"]
        text = ConfigSchema.section_text("claude_code")
        assert "claude_dir" in text and "idle_minutes" in text and "Claude Code" in text


@pytest.mark.unit
class TestOwnSessions:
    def test_the_maintainer_records_the_ids_the_feed_leaves_out(self, tmp_path, monkeypatch):
        import assistant.agents.CodeQualityMaintainer as maintainer
        own = tmp_path / "log" / "own.txt"
        monkeypatch.setattr(maintainer, "CLAUDE_OWN_SESSIONS_PATH", own)
        monkeypatch.setattr(module, "CLAUDE_OWN_SESSIONS_PATH", own)
        maintainer.CodeQualityMaintainer.record_own_session("run-1", "nightly")
        maintainer.CodeQualityMaintainer.record_own_session(None, "nightly")
        maintainer.CodeQualityMaintainer.record_own_session("run-2", "analysis")
        assert ClaudeCodeIntegration._own_sessions() == {"run-1", "run-2"}
