import time
import uuid

import pytest
from unittest.mock import Mock, call, patch

from assistant.prompt.PromptManager import PromptManager


class DummySection:
    def __init__(self, content: str, last_update_time: float = 0, section_name: str = ""):
        self._content = content
        self.last_update_time = last_update_time
        # Production generators identify themselves to PromptManager via this attribute
        # (set in PromptSectionGenerator.__init__ to display names like "Weather",
        # "CalendarData"). Tests that exercise the real notification path set it too;
        # tests that pre-date the delta routing pass empty and rely on the placeholder-
        # key fallback in on_section_updated.
        self.section_name = section_name

    def get_content(self) -> str:
        return self._content


@pytest.fixture()
def prompt_test_environment(tmp_path, monkeypatch):
    """Create an isolated working directory with the files PromptManager expects."""
    prompt_template = (
        "Header\n"
        "{PROCEDURES}\n"
        "{WATCHLIST}\n"
        "{HOME}\n"
        "{MEMORY}\n"
        "{VAULT}\n"
        "{CALENDAR}\n"
        "{INTEGRATIONS}\n"
        "{WHISPER}\n"
        "{CAR_INFO}\n"
    )
    testsuite_prompt = "Test suite prompt for Marcus"

    res_dir = tmp_path / "res"
    res_dir.mkdir()
    (res_dir / "prompt.md").write_text(prompt_template)
    (tmp_path / "testsuite_prompt.md").write_text(testsuite_prompt)

    monkeypatch.chdir(tmp_path)

    return {
        "template": prompt_template,
        "testsuite_prompt": testsuite_prompt,
        "root": tmp_path,
    }


@pytest.mark.unit
def test_current_time_note_names_the_given_instant_not_now():
    """A byte-prefix request persists the note it sent, so the note must be computable for the
    request's own start time — a Saturday afternoon here, whatever the clock says now."""
    from datetime import datetime
    at = datetime(2025, 3, 15, 14, 40, 0).timestamp()

    note = PromptManager.current_time_note(at)

    assert "14:40" in note and "Saturday" in note and "2025-03-15" in note
    assert PromptManager.current_time_note(at + 60) != note, "a minute later must render a different note"


@pytest.mark.unit
def test_a_checkout_without_the_testsuite_prompt_runs_the_suite_on_the_production_template(prompt_test_environment):
    """testsuite_prompt.md and its history fixture are private; a public checkout has neither."""
    (prompt_test_environment["root"] / "testsuite_prompt.md").unlink()

    manager = PromptManager(test_suite=True)
    manager.load_test_suite_history("res/testsuite_history.json")

    assert manager.prompt.startswith("Header\n")
    assert "{MEMORY}" not in manager.prompt and "{VAULT}" not in manager.prompt
    assert manager._test_history_messages == []


@pytest.mark.unit
def test_testsuite_prompt_loaded_and_logged(prompt_test_environment, monkeypatch, file_logging_enabled):
    test_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
    monkeypatch.setattr("assistant.prompt.PromptManager.uuid.uuid4", lambda: test_uuid)

    manager = PromptManager(test_suite=True)

    expected_prompt = prompt_test_environment["testsuite_prompt"]
    messages = manager.get_messages()

    assert manager.prompt == expected_prompt
    # The test suite sees every toolkit as one load_toolkit round after the history.
    assert messages == [{"role": "system", "content": expected_prompt}] + manager._test_toolkit_messages
    assert [m["role"] for m in manager._test_toolkit_messages] == ["assistant", "user"]
    assert manager._test_toolkit_messages[0]["content"].startswith('load_toolkit("home", "cameras", "web", ')
    assert manager._test_toolkit_messages[1]["content"].startswith("result: \n# Toolkit: home\n")
    assert "# Toolkit: timers" in manager._test_toolkit_messages[1]["content"]

    # Simulate the full messages array sent to the LLM (system + user turn)
    full_messages = messages + [{"role": "user", "content": "Hello Floxy"}]
    PromptManager.save_prompt_to_file(full_messages)

    log_file = prompt_test_environment["root"] / "log" / "prompts" / f"{test_uuid}.md"
    assert log_file.exists()
    logged = log_file.read_text()
    assert "# [SYSTEM]" in logged
    assert expected_prompt in logged
    assert "# [USER]" in logged
    assert "Hello Floxy" in logged


@pytest.mark.unit
def test_prompt_dump_goes_through_the_log_masker(prompt_test_environment, monkeypatch, file_logging_enabled):
    # The history holds real values; the dump is a log artefact and is masked like one.
    import common.logging as logging_mod
    monkeypatch.setattr(logging_mod, "_masker", lambda text: text.replace("01234567", "••••••••"))
    test_uuid = "dump-masked"
    monkeypatch.setattr("assistant.prompt.PromptManager.uuid.uuid4", lambda: test_uuid)
    PromptManager.save_prompt_to_file([{"role": "assistant", "content": 'config_set("k", "01234567")'}])

    logged = (prompt_test_environment["root"] / "log" / "prompts" / f"{test_uuid}.md").read_text()
    assert "01234567" not in logged
    assert 'config_set("k", "••••••••")' in logged


@pytest.mark.unit
def test_section_update_reassembles_prompt(prompt_test_environment):
    manager = PromptManager(test_suite=True)

    section_content = {
        "procedures": "Procedures section",
        "watchlist": "Watch list section",
        "home": "Home automation section",
        "memory": "Memory section content",
        "vault": "Vault section content",
        "calendar": "Calendar entries section",
        "integrations": "Integrations section",
        "whisper": "Whisper instructions section",
        "car_info": "Tesla car information section",
    }
    manager.sections = {key: DummySection(value) for key, value in section_content.items()}

    previous_assembly_time = manager._last_assembly_time
    manager.on_section_updated("procedures")

    expected_prompt = prompt_test_environment["template"]
    for key, value in section_content.items():
        if key != "home":  # carried by the home toolkit: never baked into a prompt
            expected_prompt = expected_prompt.replace("{" + key.upper() + "}", value)

    assert manager.prompt == expected_prompt
    assert manager._messages == [{"role": "system", "content": expected_prompt}]
    assert manager._last_assembly_time >= previous_assembly_time

    for placeholder in ("{PROCEDURES}", "{WATCHLIST}", "{MEMORY}", "{VAULT}", "{CALENDAR}", "{INTEGRATIONS}", "{WHISPER}", "{CAR_INFO}"):
        assert placeholder not in manager.prompt
    assert "{HOME}" in manager.prompt, "the fixture template's {HOME} stays: res/prompt.md has none"


# ===========================================================================
# {VAULT} section (VAULT_PLAN §3.7)
# ===========================================================================

@pytest.fixture()
def real_template_environment(tmp_path, monkeypatch):
    """Isolated cwd holding copies of the production res/prompt.md, so assembly exercises the real template.
    The same file serves as testsuite_prompt.md: in test_suite mode _compute_section_positions reads that one."""
    import shutil
    from common.Paths import AISERVER_DIR
    res_dir = tmp_path / "res"
    res_dir.mkdir()
    shutil.copy(AISERVER_DIR / "res" / "prompt.md", res_dir / "prompt.md")
    shutil.copy(AISERVER_DIR / "res" / "prompt.md", tmp_path / "testsuite_prompt.md")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(PromptManager, "_get_photo_count", lambda self: 0)
    return tmp_path


@pytest.mark.unit
def test_assembled_prompt_replaces_vault_placeholder_with_ids_only(real_template_environment, tmp_path):
    from common.vault.Vault import Vault

    manager = PromptManager(test_suite=True)
    vault = Vault(tmp_path / "vault.bin", passphrase="test")
    vault.add("phone", "Alex's mobile", "+46 70-123 45 67", id="alex-mobile")
    manager.sections["vault"]._vault = vault

    # The generator's own change detection drives the rebuild, as the monitor thread would.
    manager.sections["vault"]._refresh_if_changed()

    prompt = manager.prompt
    assert "{VAULT}" not in prompt
    assert "- {{secret:alex-mobile}} — Alex's mobile (phone)" in prompt
    assert "46701234567" not in prompt and "123 45 67" not in prompt
    # The list renders at the end of the ## Vault subsection, below the prose that introduces it.
    listed = prompt.index("- {{secret:alex-mobile}}")
    assert prompt.index("## Vault") < prompt.index("not an anonymization tool") < listed < prompt.index("## Function Calling Rules")
    assert prompt.count("{{secret:alex-mobile}}") == 3   # 2 in the docs' examples + 1 listed


@pytest.mark.unit
def test_assembled_prompt_renders_empty_vault(real_template_environment, tmp_path):
    from common.vault.Vault import Vault

    manager = PromptManager(test_suite=True)
    manager.sections["vault"]._vault = Vault(tmp_path / "vault.bin", passphrase="test")

    manager.sections["vault"]._refresh_if_changed()

    assert "{VAULT}" not in manager.prompt
    assert "not an anonymization tool.\n\nThe vault is empty.\n" in manager.prompt


# ===========================================================================
# on_prompt_assembled callback tests (KV-cache pre-heat hook)
# ===========================================================================

def _assemble_with_sections(manager, whisper_content="Whisper instructions section"):
    """Drive a real reassembly in test-suite mode and return nothing.

    Mirrors test_section_update_reassembles_prompt: with no 'history' section,
    on_section_updated forces _assemble_prompt, which fires the registered
    on_prompt_assembled callbacks.
    """
    section_content = {
        "procedures": "Procedures section",
        "watchlist": "Watch list section",
        "home": "Home automation section",
        "memory": "Memory section content",
        "vault": "Vault section content",
        "calendar": "Calendar entries section",
        "integrations": "Integrations section",
        "whisper": whisper_content,
        "car_info": "Tesla car information section",
    }
    manager.sections = {key: DummySection(value) for key, value in section_content.items()}
    manager.on_section_updated("procedures")


@pytest.mark.unit
def test_on_prompt_assembled_invokes_callback_with_assembled_messages(prompt_test_environment):
    manager = PromptManager(test_suite=True)

    received = []
    manager.on_prompt_assembled(lambda messages: received.append(messages))

    _assemble_with_sections(manager)

    assert len(received) == 1
    callback_messages = received[0]
    assert callback_messages[0]["role"] == "system"
    assert "Procedures section" in callback_messages[0]["content"]


@pytest.mark.unit
def test_on_prompt_assembled_strips_thinking_and_whisper_placeholders(prompt_test_environment):
    # The whole point of stripping [THINKING]/[WHISPER] here is that pre-heat
    # must warm the SAME prefix get_messages() ultimately sends; otherwise the
    # local-LLM KV cache never hits. If the markers leak into the callback
    # payload, the pre-heated prefix diverges from the served prompt.
    manager = PromptManager(test_suite=True)

    received = []
    manager.on_prompt_assembled(lambda messages: received.append(messages))

    _assemble_with_sections(manager, whisper_content="head [THINKING] mid [WHISPER] tail")

    content = received[0][0]["content"]
    assert "[THINKING]" not in content
    assert "[WHISPER]" not in content
    assert "head" in content and "mid" in content and "tail" in content

    # The stripped copy must not mutate the messages the server actually serves.
    assert "[THINKING]" in manager._messages[0]["content"]
    assert "[WHISPER]" in manager._messages[0]["content"]


@pytest.mark.unit
def test_on_prompt_assembled_swallows_callback_exception(prompt_test_environment):
    # A buggy pre-heat hook must never break prompt assembly, and one failing
    # callback must not starve the others.
    manager = PromptManager(test_suite=True)

    def boom(messages):
        raise RuntimeError("pre-heat backend unreachable")

    survivor = []
    manager.on_prompt_assembled(boom)
    manager.on_prompt_assembled(lambda messages: survivor.append(messages))

    _assemble_with_sections(manager)

    assert "Procedures section" in manager.prompt
    assert len(survivor) == 1


@pytest.mark.unit
def test_testsuite_mode_does_not_start_section_threads(prompt_test_environment):
    manager = PromptManager(test_suite=True)

    for section in manager.sections.values():
        assert section._thread is None  # Threads are only started in live mode


# ===========================================================================
# test-suite history fixture tests
# ===========================================================================

@pytest.mark.unit
def test_load_test_suite_history_appends_fixture_turns_to_messages(prompt_test_environment, monkeypatch):
    """Regression (ad4a5243): in test_suite mode get_messages must append the loaded
    fixture turns as message items — that is how the prompt suite now sends history,
    mirroring production. load_fixture_messages is tested in isolation, but nothing
    else guards the PromptManager wiring: drop the test_suite branch in get_messages
    and the suite would silently send no history while every unit test still passed."""
    import json
    from datetime import datetime
    from assistant.tools.MessageFormatter import MessageFormatter

    fixture = [
        {"role": "Marcus", "content": "turn off lights",
         "timestamp": "2026-05-20 14:40:00", "source": "VoiceToVoice", "location": "Kitchen"},
        {"role": "assistant", "content": "Done.", "timestamp": "2026-05-20 14:40:05"},
    ]
    path = prompt_test_environment["root"] / "testsuite_history.json"
    path.write_text(json.dumps(fixture))

    manager = PromptManager(test_suite=True)
    manager.load_test_suite_history(str(path))

    messages = manager.get_messages()

    ts = datetime(2026, 5, 20, 14, 40, 0).timestamp()
    assert messages[0]["role"] == "system"
    assert messages[1] == {
        "role": "user",
        "content": MessageFormatter.format_message(
            "Marcus", ts, "VoiceToVoice", "Kitchen", "turn off lights"),
    }
    assert messages[2] == {"role": "assistant", "content": "Done."}


@pytest.mark.unit
def test_test_suite_fixture_history_precedes_thinking_turn(prompt_test_environment, monkeypatch):
    """The extended-thinking user turn must land after the fixture history, not before
    it — otherwise the model sees the thinking instruction out of order. Confirms the
    fixture branch and the thinking-append path compose correctly in test_suite mode."""
    import json

    fixture = [
        {"role": "Marcus", "content": "what's the weather",
         "timestamp": "2026-05-20 09:15:00", "source": None, "location": None},
        {"role": "assistant", "content": "Sunny.", "timestamp": "2026-05-20 09:15:03"},
    ]
    path = prompt_test_environment["root"] / "testsuite_history.json"
    path.write_text(json.dumps(fixture))

    manager = PromptManager(test_suite=True)
    manager.load_test_suite_history(str(path))

    messages = manager.get_messages(extended_thinking=True)

    # After the fixture comes the test suite's toolkit load round; the thinking instruction
    # merges into its result turn, the trailing user turn, as it would into any user turn.
    assert messages[-3] == {"role": "assistant", "content": "Sunny."}
    assert messages[-2] == manager._test_toolkit_messages[0]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].startswith(manager._test_toolkit_messages[1]["content"])
    assert "<thinking>" in messages[-1]["content"]


# ===========================================================================
# update_history tests
# ===========================================================================

# ===========================================================================
# get_messages thinking/whisper tests
# ===========================================================================

@pytest.mark.unit
def test_get_messages_with_extended_thinking(prompt_test_environment, monkeypatch):
    monkeypatch.setattr("assistant.prompt.PromptManager.uuid.uuid4", lambda: uuid.uuid4())
    manager = PromptManager(test_suite=True)

    manager._messages = [{"role": "system", "content": "Before [THINKING] after [WHISPER] end"}]

    from assistant.prompt.TimeOfDay import TimeOfDay
    monkeypatch.setattr('assistant.prompt.PromptManager.TimeOfDay.get_current', lambda: TimeOfDay.DAY)

    messages = manager.get_messages(extended_thinking=True)

    content = messages[0]["content"]
    assert "[THINKING]" not in content
    assert "[WHISPER]" not in content
    assert len(messages) == 3      # system, the test suite's toolkit load call, its result + the thinking turn
    assert messages[-1]["role"] == "user"
    instruction = messages[-1]["content"]
    assert "<thinking>" in instruction
    # Regression for d45d4718: the instruction must (a) keep the door open for
    # skipping the tag on simple queries — otherwise short replies become
    # latency-heavy — and (b) demand that ANY reasoning Claude *does* produce
    # stays inside the tag, since _remove_thinking() only strips tagged
    # content. Without (b), untagged preambles were getting spoken aloud.
    low = instruction.lower()
    assert "skip" in low, "instruction must let the LLM skip the tag on simple queries"
    assert "all" in low and "reasoning" in low, \
        "instruction must require ALL reasoning to live inside the tag"


@pytest.mark.unit
def test_get_messages_without_extended_thinking_never_mentions_the_tag(prompt_test_environment, monkeypatch):
    # What a model that thinks on its own receives (RequestProcessor passes False for it):
    # Claude refuses the whole request as reasoning extraction if the tag shows up anywhere.
    manager = PromptManager(test_suite=True)
    manager._messages = [{"role": "system", "content": "Before [THINKING] after"}]

    from assistant.prompt.TimeOfDay import TimeOfDay
    monkeypatch.setattr('assistant.prompt.PromptManager.TimeOfDay.get_current', lambda: TimeOfDay.DAY)

    messages = manager.get_messages(extended_thinking=False)

    assert "[THINKING]" not in messages[0]["content"]
    assert not [m for m in messages if "<thinking" in str(m.get("content")).lower()]


@pytest.mark.unit
def test_static_prompts_never_ask_for_a_thinking_tag():
    # The templates and toolkits reach every model unconditionally, Claude included, so the
    # tag instruction may only ever be added per model in code (handles_thinking_internally).
    from common.Paths import RES_DIR
    prompts = list(RES_DIR.rglob("*.md"))
    assert prompts
    assert [p.name for p in prompts if "<thinking" in p.read_text(encoding="utf-8").lower()] == []


@pytest.mark.unit
def test_get_messages_night_whisper(prompt_test_environment, monkeypatch):
    monkeypatch.setattr("assistant.prompt.PromptManager.uuid.uuid4", lambda: uuid.uuid4())
    manager = PromptManager(test_suite=True)

    manager._messages = [{"role": "system", "content": "Prompt [WHISPER] end [THINKING] done"}]

    from assistant.prompt.TimeOfDay import TimeOfDay
    monkeypatch.setattr('assistant.prompt.PromptManager.TimeOfDay.get_current', lambda: TimeOfDay.NIGHT)

    messages = manager.get_messages(extended_thinking=False)

    content = messages[0]["content"]
    assert "[WHISPER]" not in content
    assert "whisper" in content.lower()
    assert "[THINKING]" not in content


@pytest.mark.unit
def test_get_messages_daytime_no_whisper(prompt_test_environment, monkeypatch):
    monkeypatch.setattr("assistant.prompt.PromptManager.uuid.uuid4", lambda: uuid.uuid4())
    manager = PromptManager(test_suite=True)

    manager._messages = [{"role": "system", "content": "Prompt [WHISPER] end [THINKING] done"}]

    from assistant.prompt.TimeOfDay import TimeOfDay
    monkeypatch.setattr('assistant.prompt.PromptManager.TimeOfDay.get_current', lambda: TimeOfDay.DAY)

    messages = manager.get_messages(extended_thinking=False)

    content = messages[0]["content"]
    assert "[WHISPER]" not in content
    assert "whisper" not in content.lower()


@pytest.mark.unit
def test_get_messages_merges_thinking_into_trailing_user_history(prompt_test_environment, monkeypatch):
    # Regression: when the most recent history turn maps to `user` (an unanswered
    # incoming message or a system event), the always-on thinking instruction must
    # merge into it rather than append a second consecutive user turn — Gemini and
    # other providers reject same-role adjacency.
    manager = PromptManager(test_suite=True)
    manager.test_suite = False  # enable the live-history append path in get_messages

    mock_history = Mock()

    mock_history.dropped_delta_sections = set()
    mock_history.get_history_messages.return_value = [
        {"role": "user", "content": "Bob (iMessage): you up?"},
    ]
    manager.message_history = mock_history
    manager._messages = [{"role": "system", "content": "system prompt"}]

    from assistant.prompt.TimeOfDay import TimeOfDay
    monkeypatch.setattr('assistant.prompt.PromptManager.TimeOfDay.get_current', lambda: TimeOfDay.DAY)

    messages = manager.get_messages(extended_thinking=True)

    roles = [m["role"] for m in messages]
    assert all(a != b for a, b in zip(roles, roles[1:])), f"consecutive same-role turns: {roles}"
    assert messages[-1]["role"] == "user"
    assert "you up?" in messages[-1]["content"]
    assert "<thinking>" in messages[-1]["content"]


@pytest.mark.unit
def test_get_messages_keeps_thinking_separate_after_assistant_history(prompt_test_environment, monkeypatch):
    # Common case: history ends in an assistant reply, so the thinking instruction
    # is its own trailing user turn. The assistant turn stays as messages[-2],
    # which is where ClaudeLLM places the cache breakpoint — full history cached.
    manager = PromptManager(test_suite=True)
    manager.test_suite = False

    mock_history = Mock()

    mock_history.dropped_delta_sections = set()
    mock_history.get_history_messages.return_value = [
        {"role": "user", "content": "what time is it?"},
        {"role": "assistant", "content": "3pm"},
    ]
    manager.message_history = mock_history
    manager._messages = [{"role": "system", "content": "system prompt"}]

    from assistant.prompt.TimeOfDay import TimeOfDay
    monkeypatch.setattr('assistant.prompt.PromptManager.TimeOfDay.get_current', lambda: TimeOfDay.DAY)

    messages = manager.get_messages(extended_thinking=True)

    assert messages[-1]["role"] == "user"
    assert "<thinking>" in messages[-1]["content"]
    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["content"] == "3pm"


@pytest.mark.unit
def test_get_messages_appends_current_time_note_in_production_mode(prompt_test_environment, monkeypatch):
    # Regression for b8cb3ed7: time/date used to live in the system prompt as
    # {TIME_DATE} but rewrote the prompt bytes every minute, invalidating the
    # 5-min message-level KV cache on every tick. It now rides on the trailing
    # user turn — uncached anyway, so zero cache impact. The format
    # "(Current time: HH:MM, Weekday, YYYY-MM-DD)" is what the LLM has been
    # trained against; a silent format drift would degrade time-aware answers
    # without anything failing loudly.
    from datetime import datetime

    manager = PromptManager(test_suite=True)
    manager.test_suite = False

    mock_history = Mock()

    mock_history.dropped_delta_sections = set()
    mock_history.get_history_messages.return_value = [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "ok"},
    ]
    manager.message_history = mock_history
    manager._messages = [{"role": "system", "content": "system prompt"}]

    from assistant.prompt.TimeOfDay import TimeOfDay
    monkeypatch.setattr('assistant.prompt.PromptManager.TimeOfDay.get_current', lambda: TimeOfDay.DAY)

    # Freeze wall-clock so the assertion can be exact. Monday 2026-05-25 14:07.
    frozen = datetime(2026, 5, 25, 14, 7, 0)
    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen
    monkeypatch.setattr('assistant.prompt.PromptManager.datetime', _Frozen)

    messages = manager.get_messages(extended_thinking=False)

    # Without thinking, the time note is the only thing appended after history,
    # so it lives in its own trailing user turn (history ended in assistant).
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "(Current time: 14:07, Monday, 2026-05-25)"


@pytest.mark.unit
def test_get_messages_skips_time_note_in_test_suite_mode(prompt_test_environment, monkeypatch):
    # Contract for unit tests: test_suite=True must keep get_messages() free of
    # wall-clock state so suites stay deterministic. If the skip ever broke,
    # every prompt/integration test that snapshots messages would silently flake.
    manager = PromptManager(test_suite=True)
    manager._messages = [{"role": "system", "content": "system prompt"}]

    from assistant.prompt.TimeOfDay import TimeOfDay
    monkeypatch.setattr('assistant.prompt.PromptManager.TimeOfDay.get_current', lambda: TimeOfDay.DAY)

    messages = manager.get_messages(extended_thinking=False)

    joined = "\n".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
    assert "Current time:" not in joined


# ===========================================================================
# is_night_mode_active tests (the live home snapshot, NOT the frozen prompt)
# ===========================================================================
#
# Regression: the autonomous-action night guards used to scan self.prompt for
# '"Night": true'. With the cache-stable refactor, a Night flip arriving as a
# small section delta freezes the prompt bytes (only a transient note is
# emitted), so self.prompt lags the real Night state by up to the 1h rebuild
# interval. is_night_mode_active() must read the LIVE home section instead.


def _home_section(night: str):
    """A real HomeSectionGenerator with its live snapshot set directly (no SSH/thread)."""
    from assistant.integration.HomeAutomation import Entity, HomeSnapshot
    from assistant.prompt.sections.HomeSectionGenerator import HomeSectionGenerator
    section = HomeSectionGenerator(prompt_manager=None)
    section._latest = HomeSnapshot(variables=[Entity("Night", night)])
    return section


@pytest.mark.unit
def test_is_night_mode_active_reads_live_section_not_stale_prompt(prompt_test_environment):
    manager = PromptManager(test_suite=True)
    # The live snapshot says Night; the frozen prompt snapshot does NOT contain the flag.
    manager.sections['home'] = _home_section("true")
    manager.prompt = "frozen system prompt with no night flag"

    assert manager.is_night_mode_active() is True, \
        "must reflect live section state even when the frozen prompt lacks the flag"


@pytest.mark.unit
def test_is_night_mode_active_false_when_live_section_clear_despite_stale_prompt(prompt_test_environment):
    manager = PromptManager(test_suite=True)
    # The live snapshot says day; the stale frozen prompt still literally contains the flag.
    manager.sections['home'] = _home_section("false")
    manager.prompt = 'stale snapshot "Night": true'

    assert manager.is_night_mode_active() is False, \
        "stale '\"Night\": true' in the frozen prompt must not register as Night"


@pytest.mark.unit
def test_is_night_mode_active_false_when_section_raises(prompt_test_environment):
    manager = PromptManager(test_suite=True)
    boom = Mock()
    boom.is_night.side_effect = RuntimeError("section unavailable")
    manager.sections['home'] = boom
    # A failing read must default to "not Night" (fail-open: better to run a check
    # than to silently suppress autonomous action forever on a transient error).
    assert manager.is_night_mode_active() is False


# ===========================================================================
# current_time_note tests
# ===========================================================================

@pytest.mark.unit
def test_current_time_note_format(prompt_test_environment, monkeypatch):
    # Single source of truth for the clock note shared by get_messages() and the
    # autonomous-action check. The exact format is what the LLM is trained against.
    from datetime import datetime

    frozen = datetime(2026, 5, 29, 14, 50, 0)  # Friday

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen

    monkeypatch.setattr('assistant.prompt.PromptManager.datetime', _Frozen)

    assert PromptManager.current_time_note() == "(Current time: 14:50, Friday, 2026-05-29)"


# ===========================================================================
# get_section_stats tests
# ===========================================================================

@pytest.mark.unit
def test_get_section_stats_ordered_by_template_position(tmp_path, monkeypatch):
    template = "{CALENDAR}\n{PROCEDURES}\n{HOME}\n"
    testsuite = template
    res_dir = tmp_path / "res"
    res_dir.mkdir()
    (res_dir / "prompt.md").write_text(template)
    (tmp_path / "testsuite_prompt.md").write_text(testsuite)
    monkeypatch.chdir(tmp_path)

    manager = PromptManager(test_suite=True)
    manager.sections = {
        'procedures': DummySection("w"),
        'home': DummySection("i"),
        'calendar': DummySection("c"),
    }
    manager._section_positions = manager._compute_section_positions()
    manager.prompt = "cwi"

    stats = manager.get_section_stats()
    names = [s[0] for s in stats if s[0] not in ('static', 'history')]

    assert names == ['calendar', 'procedures', 'home']


@pytest.mark.unit
def test_get_section_stats_leaves_out_a_section_without_a_placeholder(tmp_path, monkeypatch):
    """The Indigo data is carried by the home toolkit and never in the prompt: no bar for
    it, and its bytes must not be subtracted from the static size (the dashboard's static
    bar went negative and the Indigo bar set the scale after {HOME} left prompt.md)."""
    template = "{PROCEDURES}\n"
    res_dir = tmp_path / "res"
    res_dir.mkdir()
    (res_dir / "prompt.md").write_text(template)
    (tmp_path / "testsuite_prompt.md").write_text(template)
    monkeypatch.chdir(tmp_path)

    manager = PromptManager(test_suite=True)
    manager.sections = {
        'home': DummySection("x" * 30_000),
        'procedures': DummySection("w"),
    }
    manager._section_positions = manager._compute_section_positions()
    manager.prompt = "static w"

    stats = {name: size for name, size, _ in manager.get_section_stats()}

    assert 'home' not in stats
    assert stats['procedures'] == 1 and stats['static'] == len("static ")


@pytest.mark.unit
def test_get_section_stats_without_a_readable_template_lists_every_section(tmp_path, monkeypatch):
    """No template means no positions at all — then nothing can be told apart and every
    section keeps its bar, alphabetically."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "testsuite_prompt.md").write_text("")
    manager = PromptManager(test_suite=True)
    manager.sections = {'zebra': DummySection("z"), 'alpha': DummySection("a")}
    manager._section_positions = {}
    manager.prompt = "za"

    names = [s[0] for s in manager.get_section_stats() if s[0] not in ('static', 'history')]
    assert names == ['alpha', 'zebra']


@pytest.mark.unit
def test_get_section_stats_history_sized_from_llm_payload(prompt_test_environment):
    """Regression (c5c24ef8): the history bar must report the bytes actually sent to
    the LLM — the rendered message turns from get_history_messages() — not some other
    rendering of the same entries (the console text view it once used formatted turns
    differently, so the bar was a proxy, not the real payload)."""
    manager = PromptManager(test_suite=True)
    manager.sections = {}
    manager.prompt = "static"
    manager.message_history.history = [
        {"role": "Marcus", "content": "turn off the lights", "timestamp": 1000.0,
         "source": "VoiceToVoice", "location": "Kitchen"},
        {"role": "assistant", "content": "Done.", "timestamp": 1001.0},
    ]

    history_size = next(size for name, size, _ in manager.get_section_stats() if name == 'history')

    expected = sum(len(m["content"].encode("utf-8"))
                   for m in manager.message_history.get_history_messages())
    assert history_size == expected
    assert history_size > 0


@pytest.mark.unit
def test_compute_section_positions_returns_empty_on_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    manager = PromptManager.__new__(PromptManager)
    manager.test_suite = True
    manager.sections = {'procedures': DummySection("w")}

    positions = manager._compute_section_positions()

    assert positions == {}


# ===========================================================================
# stop tests
# ===========================================================================

@pytest.mark.unit
def test_stop_calls_stop_on_sections(prompt_test_environment):
    manager = PromptManager(test_suite=True)

    stoppable = Mock()
    stoppable.stop = Mock()
    non_stoppable = DummySection("content")

    manager.sections = {
        'a': stoppable,
        'b': non_stoppable,
    }

    manager.stop()

    stoppable.stop.assert_called_once()


# ===========================================================================
# _save_image_block tests
# ===========================================================================

@pytest.mark.unit
def test_save_image_block_jpeg(tmp_path, monkeypatch, file_logging_enabled):
    import base64
    monkeypatch.chdir(tmp_path)

    raw_bytes = b"\xff\xd8\xff\xe0fake_jpeg_data"
    b64_data = base64.b64encode(raw_bytes).decode()
    block = {"source": {"data": b64_data, "media_type": "image/jpeg"}}

    path = PromptManager._save_image_block(block, "abc123", 0)

    assert path is not None
    assert path.endswith(".jpg")
    assert (tmp_path / path).exists()
    assert (tmp_path / path).read_bytes() == raw_bytes


@pytest.mark.unit
def test_save_image_block_png(tmp_path, monkeypatch, file_logging_enabled):
    import base64
    monkeypatch.chdir(tmp_path)

    raw_bytes = b"\x89PNG\r\n\x1a\nfake_png_data"
    b64_data = base64.b64encode(raw_bytes).decode()
    block = {"source": {"data": b64_data, "media_type": "image/png"}}

    path = PromptManager._save_image_block(block, "abc123", 1)

    assert path.endswith(".png")
    assert (tmp_path / path).read_bytes() == raw_bytes


@pytest.mark.unit
def test_save_image_block_returns_none_when_logging_disabled(monkeypatch):
    monkeypatch.setattr("common.logging.disable_file_logging", True)

    block = {"source": {"data": "some_data", "media_type": "image/jpeg"}}
    result = PromptManager._save_image_block(block, "abc", 0)

    assert result is None


@pytest.mark.unit
def test_save_image_block_returns_none_on_bad_base64(tmp_path, monkeypatch, file_logging_enabled):
    monkeypatch.chdir(tmp_path)

    block = {"source": {"data": "!!!not_valid_base64!!!", "media_type": "image/jpeg"}}
    result = PromptManager._save_image_block(block, "abc", 0)

    assert result is None


# ===========================================================================
# Section delta / transient-rebuild tests (KV cache preservation)
# ===========================================================================
#
# The whole point of these tests is the cost-driver fix: small section changes
# must NOT rebuild the system prompt (which would invalidate Anthropic's 1h
# ephemeral cache). Instead they emit a transient "system prompt diff" note
# into MessageHistory. Large changes / expired-window / missing-baseline must
# still trigger a full rebuild.


def _seed_full_rebuild(prompt_test_environment):
    """Build a PromptManager that's gone through one full rebuild — baselines
    snapshotted, transients cleared, _last_assembly_time set to 'now'.

    test_suite=True so we don't spin up the real section threads; we then swap
    in DummySections matching the template placeholders. The first
    on_section_updated call with a missing baseline goes through the cold-start
    full-rebuild path, which is what we want as the setup."""
    manager = PromptManager(test_suite=True)
    sections = {key: DummySection(f"{key}_content") for key in
                ["procedures", "watchlist", "home", "memory",
                 "calendar", "integrations", "whisper", "car_info"]}
    manager.sections = sections
    manager._last_assembly_time = 0
    manager.on_section_updated("procedures")  # cold start → full rebuild snapshots baselines
    return manager, sections


@pytest.mark.unit
def test_small_change_emits_transient_and_skips_rebuild(prompt_test_environment):
    manager, sections = _seed_full_rebuild(prompt_test_environment)
    received = []
    manager.on_prompt_assembled(lambda msgs: received.append(msgs))

    prompt_before = manager.prompt
    baselines_before = dict(manager._section_baselines)

    sections["procedures"] = DummySection("procedures_content_x")  # ~1-char addition
    manager.sections = sections
    manager.on_section_updated("procedures")

    assert manager.prompt == prompt_before, "small change must not rebuild the prompt"
    assert manager._section_baselines == baselines_before, "baselines must not be re-snapshotted on delta"
    assert len(received) == 1, "pre-heat callback must fire on a delta note too — it extends the request prefix"
    assert len(manager.message_history.transient_entries) == 1
    note = manager.message_history.transient_entries[0]
    assert note["content"].startswith("Prompt section `procedures` updated:")
    assert note["source"] == "prompt_delta"


@pytest.mark.unit
def test_update_history_fires_pre_heat_with_the_new_exchange(prompt_test_environment):
    """A completed turn extends the request prefix just like a delta note does —
    the pre-heat must fire so the next query doesn't prefill the previous
    exchange at query time. test_suite mode substitutes a fixture history, so
    flip the flag to exercise the live-history path."""
    manager, sections = _seed_full_rebuild(prompt_test_environment)
    manager.test_suite = False
    # Mock the history so the test doesn't depend on the FloxyDatabase
    # singleton (shared across tests, re-homed by other fixtures).
    rendered_turn = {"role": "user", "content": "User (20:30): What time is it?\n\nFloxy (20:30): It's 3pm"}
    manager.message_history = Mock()
    manager.message_history.dropped_delta_sections = set()
    manager.message_history.get_history_messages.return_value = [rendered_turn]
    received = []
    manager.on_prompt_assembled(lambda msgs: received.append(msgs))

    manager.update_history("Alice", "What time is it?", "It's 3pm", source="voice")

    manager.message_history.add_exchange.assert_called_once_with(
        "Alice", "What time is it?", "It's 3pm", "voice", None, timestamp=None, note=None)
    assert len(received) == 1
    assert received[0][0]["role"] == "system"
    assert received[0][1:] == [rendered_turn], "pre-heat payload must include the rendered exchange"


@pytest.mark.unit
def test_update_history_pre_heat_false_records_exchange_without_pre_heating(prompt_test_environment):
    """Command follow-up rounds record the intermediate exchange with pre_heat=False:
    the follow-up inference starts immediately after and would race the pre-heat on
    Ollama's single slot. The exchange must still land in history."""
    manager, sections = _seed_full_rebuild(prompt_test_environment)
    manager.test_suite = False
    manager.message_history = Mock()
    manager.message_history.dropped_delta_sections = set()
    received = []
    manager.on_prompt_assembled(lambda msgs: received.append(msgs))

    with patch.object(manager, '_resolved_messages_locked') as mock_build:
        manager.update_history("Alice", "Turn on the lights", "On it", source="voice",
                               pre_heat=False)

    manager.message_history.add_exchange.assert_called_once_with(
        "Alice", "Turn on the lights", "On it", "voice", None, timestamp=None, note=None)
    assert received == [], "mid-request history update must not fire the pre-heat"
    mock_build.assert_not_called()


@pytest.mark.unit
def test_update_history_skips_pre_heat_build_without_callbacks(prompt_test_environment):
    """No registered callbacks (test suite, LLMs without pre-heat) — update_history
    must not spend time building the resolved messages."""
    manager, sections = _seed_full_rebuild(prompt_test_environment)
    with patch.object(manager, '_resolved_messages_locked') as mock_build:
        manager.update_history("Alice", "hi", "hello")
    mock_build.assert_not_called()


@pytest.mark.unit
def test_delta_pre_heat_messages_include_the_transient_note(prompt_test_environment):
    """The whole point of pre-heating on the delta path: the warmed bytes must
    include the freshly appended note, i.e. the exact prefix of the next real
    request. test_suite mode substitutes a fixture history in get_messages, so
    flip the flag after seeding to exercise the live-history path."""
    manager, sections = _seed_full_rebuild(prompt_test_environment)
    manager.test_suite = False
    received = []
    manager.on_prompt_assembled(lambda msgs: received.append(msgs))

    sections["procedures"] = DummySection("procedures_content_x")
    manager.sections = sections
    manager.on_section_updated("procedures")

    assert len(received) == 1
    messages = received[0]
    assert messages[0]["role"] == "system"
    rendered_tail = "".join(m["content"] for m in messages[1:])
    assert "procedures" in rendered_tail, "pre-heat payload must contain the delta note"


def _send(manager):
    """A request goes out: get_messages() seals the notes it carries, so the next change to
    their section adds a note of its own instead of folding into them."""
    manager.get_messages()


def _change(manager, sections, key, content):
    sections[key] = DummySection(content)
    manager.sections = sections
    manager.on_section_updated(key)


def _trim_away_the_notes(manager):
    """Make the history window evict every transient note: date the notes before a run of
    large turns and set the watermarks so the trim cuts between them. The returned
    messages are what the next request would send."""
    now = time.time()
    for entry in manager.message_history.transient_entries:
        entry["timestamp"] = now - 100
    manager.message_history.history = [
        {"role": "Marcus", "content": f"m{i:03d}" + "x" * 100, "timestamp": now - 50 + i,
         "source": None, "location": None}
        for i in range(30)
    ]
    manager.message_history.trim_high_bytes = 2_000
    manager.message_history.trim_low_bytes = 1_000
    manager.message_history.trim_ceiling_bytes = 2_000   # the rows are fresh; the test is about the trim's effect
    with manager._prompt_lock:
        return manager._resolved_messages_locked()


@pytest.mark.unit
def test_a_trim_that_evicts_the_delta_notes_leaves_one_note_restating_the_section(prompt_test_environment):
    """The live failure: an hour-old system prompt said the car was driving, the notes that
    said it had parked were trimmed out by a burst of browser output, and the model
    reported the driver mid-commute. After such a trim the window must carry one note
    diffing the baseline against the section's current state — not the chain of
    intermediate notes, and not nothing."""
    manager, sections = _seed_full_rebuild(prompt_test_environment)
    manager.test_suite = False
    for state in ("procedures_rain", "procedures_sun"):
        _change(manager, sections, "procedures", state)
        _send(manager)
    assert len(manager.message_history.transient_entries) == 2

    messages = _trim_away_the_notes(manager)

    notes = [e for e in manager.message_history.transient_entries if e.get("section") == "procedures"]
    assert len(notes) == 1
    assert "-procedures_content" in notes[0]["content"] and "+procedures_sun" in notes[0]["content"]
    assert "procedures_rain" not in notes[0]["content"], "the intermediate state is history, not current"
    rendered = "".join(m["content"] for m in messages[1:])
    assert "+procedures_sun" in rendered, "the fresh note must be inside the window the request sends"
    assert manager.message_history.dropped_delta_sections == set()


@pytest.mark.unit
def test_a_section_back_at_its_baseline_gets_no_note_after_a_trim(prompt_test_environment):
    """Baseline → A → baseline: the system prompt is right again, so restating would only
    tell the LLM about a change that is not there."""
    manager, sections = _seed_full_rebuild(prompt_test_environment)
    manager.test_suite = False
    for state in ("procedures_rain", "procedures_content"):
        sections["procedures"] = DummySection(state)
        manager.sections = sections
        manager.on_section_updated("procedures")

    messages = _trim_away_the_notes(manager)

    assert not [e for e in manager.message_history.transient_entries if e.get("section") == "procedures"]
    assert "procedures_rain" not in "".join(m["content"] for m in messages[1:])


@pytest.mark.unit
def test_large_change_forces_full_rebuild_and_keeps_the_notes(prompt_test_environment):
    manager, sections = _seed_full_rebuild(prompt_test_environment)
    received = []
    manager.on_prompt_assembled(lambda msgs: received.append(msgs))

    # A note already in the window: the rebuild bakes its content into the prompt but the
    # row stays — dropping it would change the window's bytes and lose the prefix cache.
    manager.message_history.add_transient_entry("system", "old note", 1000.0, source="prompt_delta")
    window_before = manager.message_history.get_history_messages()

    huge = "X" * (PromptManager._DIFF_SIZE_THRESHOLD * 5)
    sections["calendar"] = DummySection(huge)
    manager.sections = sections
    manager.on_section_updated("calendar")

    assert huge in manager.prompt, "large change must be baked into the prompt"
    assert manager._section_baselines["calendar"] == huge, "baselines must be re-snapshotted"
    assert len(received) == 1, "pre-heat callback must fire on full rebuild"
    assert manager.message_history.get_history_messages() == window_before, \
        "a full rebuild changes the prompt, never the window"


@pytest.mark.unit
def test_expired_rebuild_window_forces_full_rebuild_even_for_small_change(prompt_test_environment, monkeypatch):
    manager, sections = _seed_full_rebuild(prompt_test_environment)
    received = []
    manager.on_prompt_assembled(lambda msgs: received.append(msgs))

    # Pretend we are well past the rebuild window since baselines were snapshotted.
    manager._last_assembly_time = time.time() - (PromptManager._FULL_REBUILD_INTERVAL_S + 10)

    sections["procedures"] = DummySection("procedures_content_x")  # would otherwise be a tiny delta
    manager.sections = sections
    manager.on_section_updated("procedures")

    assert "procedures_content_x" in manager.prompt, "expired window must trigger full rebuild"
    assert len(received) == 1
    assert manager.message_history.transient_entries == []


@pytest.mark.unit
def test_missing_baseline_forces_full_rebuild(prompt_test_environment):
    """If a section has no baseline (cold start, or a brand-new section added after
    init), we can't safely diff — fall back to full rebuild."""
    manager = PromptManager(test_suite=True)
    sections = {key: DummySection(f"{key}_content") for key in
                ["procedures", "watchlist", "home", "memory",
                 "calendar", "integrations", "whisper", "car_info"]}
    manager.sections = sections
    received = []
    manager.on_prompt_assembled(lambda msgs: received.append(msgs))

    assert manager._section_baselines == {}
    manager._last_assembly_time = time.time()  # recent — so it's NOT the expiry path that triggers

    manager.on_section_updated("procedures")

    assert "procedures_content" in manager.prompt
    assert manager._section_baselines["procedures"] == "procedures_content"
    assert len(received) == 1


@pytest.mark.unit
def test_system_prompt_bytes_stable_across_many_small_updates(prompt_test_environment):
    """The whole reason for this feature: drive many small section updates in a row
    and verify the system prompt bytes are byte-identical to the post-rebuild snapshot.
    That stability is what keeps Anthropic's 1h cache warm in production."""
    manager, sections = _seed_full_rebuild(prompt_test_environment)
    prompt_after_assembly = manager.prompt

    for i, key in enumerate(["procedures", "calendar", "memory", "integrations", "car_info"]):
        sections[key] = DummySection(f"{key}_content_v{i}")
        manager.sections = sections
        manager.on_section_updated(key)

    assert manager.prompt == prompt_after_assembly, "system prompt bytes must not drift across small updates"
    # All five updates emitted a transient note.
    assert len(manager.message_history.transient_entries) == 5
    assert all(e["source"] == "prompt_delta" for e in manager.message_history.transient_entries)


@pytest.mark.unit
def test_repeated_small_change_same_section_appends_incremental_notes(prompt_test_environment):
    """Append-only changelog: when one section updates several times in a row
    (v0 → v1 → v2) each change appends its OWN note diffing against the immediately
    preceding state — earlier notes are KEPT (not superseded), so the LLM sees an
    accurate, separately-timestamped history of each change rather than one coalesced
    diff re-stating every change under the latest timestamp. That holds for notes a request
    has sent; unsent ones fold together (test_unsent_changes_to_one_section_fold_into_one_note)."""
    manager, sections = _seed_full_rebuild(prompt_test_environment)

    for i in range(3):
        _change(manager, sections, "procedures", f"procedures_content_v{i}")
        _send(manager)

    entries = manager.message_history.transient_entries
    assert len(entries) == 3, "each change appends its own incremental note"
    assert all(e["section"] == "procedures" for e in entries)
    # First note diffs against the baseline; each later note diffs against the prev state.
    assert "+procedures_content_v0" in entries[0]["content"]
    assert "-procedures_content_v0" in entries[1]["content"]
    assert "+procedures_content_v1" in entries[1]["content"]
    assert "-procedures_content_v1" in entries[2]["content"]
    assert "+procedures_content_v2" in entries[2]["content"]
    # Last-seen state advanced to the latest content for the next diff.
    assert manager._section_current["procedures"] == "procedures_content_v2"


@pytest.mark.unit
def test_section_revert_to_baseline_emits_honest_incremental_delta(prompt_test_environment):
    """'baseline → A → baseline' case: a section changes (emitting a delta), then
    reverts to exactly the baseline bytes. With incremental, append-only notes the
    revert is just another honest delta (A → baseline) stamped at the moment it
    happened — the earlier 'changed to A' note stays, because it accurately records
    that the section WAS A for a while. The LLM reads current state from the latest
    note's +side (back to baseline)."""
    manager, sections = _seed_full_rebuild(prompt_test_environment)

    sections["procedures"] = DummySection("procedures_content_changed")
    manager.sections = sections
    manager.on_section_updated("procedures")
    assert len(manager.message_history.transient_entries) == 1, "precondition: one delta emitted"
    _send(manager)

    # Revert to the exact baseline content.
    sections["procedures"] = DummySection("procedures_content")
    manager.sections = sections
    manager.on_section_updated("procedures")

    entries = manager.message_history.transient_entries
    assert len(entries) == 2, "the revert appends its own delta; the earlier note is kept"
    assert "+procedures_content_changed" in entries[0]["content"]
    assert "-procedures_content_changed" in entries[1]["content"]
    assert "+procedures_content" in entries[1]["content"]
    # Last-seen state is back to the baseline content.
    assert manager._section_current["procedures"] == "procedures_content"


@pytest.mark.unit
def test_distinct_sections_keep_independent_incremental_notes(prompt_test_environment):
    """Updates to different sections stay independent: procedures's two changes append
    two procedures notes (each diffing against the prior procedures state) while calendar's
    single change appends its own note — three notes total, none clobbering another."""
    manager, sections = _seed_full_rebuild(prompt_test_environment)

    sections["procedures"] = DummySection("procedures_content_a")
    manager.sections = sections
    manager.on_section_updated("procedures")
    _send(manager)

    sections["procedures"] = DummySection("procedures_content_b")
    manager.sections = sections
    manager.on_section_updated("procedures")

    sections["calendar"] = DummySection("calendar_content_x")
    manager.sections = sections
    manager.on_section_updated("calendar")

    entries = manager.message_history.transient_entries
    assert len(entries) == 3
    procedures_notes = [e for e in entries if e["section"] == "procedures"]
    calendar_notes = [e for e in entries if e["section"] == "calendar"]
    assert len(procedures_notes) == 2 and len(calendar_notes) == 1
    # procedures: first note baseline→a, second note a→b.
    assert "+procedures_content_a" in procedures_notes[0]["content"]
    assert "-procedures_content_a" in procedures_notes[1]["content"]
    assert "+procedures_content_b" in procedures_notes[1]["content"]
    assert "+calendar_content_x" in calendar_notes[0]["content"]


@pytest.mark.unit
def test_unsent_changes_to_one_section_fold_into_one_note(prompt_test_environment):
    """A car driving to work changes its line every minute; with no request in between, the
    notes since the last reply ride uncached in every autonomous check. They fold into one
    note from the state before the first to the latest, stamped at the latest change."""
    manager, sections = _seed_full_rebuild(prompt_test_environment)

    for street in ("Kungsgatan", "Hisingsbron", "Burggrevegatan"):
        _change(manager, sections, "car_info", f"Driving on {street}")

    entries = manager.message_history.transient_entries
    assert len(entries) == 1
    assert entries[0]["content"].endswith("-car_info_content\n+Driving on Burggrevegatan")
    assert manager._section_current["car_info"] == "Driving on Burggrevegatan"


@pytest.mark.unit
def test_a_sent_note_stays_and_the_next_change_adds_its_own(prompt_test_environment):
    manager, sections = _seed_full_rebuild(prompt_test_environment)
    _change(manager, sections, "car_info", "Driving on Kungsgatan")
    _send(manager)

    _change(manager, sections, "car_info", "Parked on Burggrevegatan")
    _change(manager, sections, "car_info", "Parked on Burggrevegatan, locked")

    notes = [e["content"] for e in manager.message_history.transient_entries]
    assert len(notes) == 2
    assert notes[0].endswith("-car_info_content\n+Driving on Kungsgatan"), "the sent note keeps its bytes"
    assert notes[1].endswith("-Driving on Kungsgatan\n+Parked on Burggrevegatan, locked"), \
        "the unsent one folds from where the sent one ended"


@pytest.mark.unit
def test_an_assistant_reply_after_a_note_keeps_it(prompt_test_environment):
    """The autonomous check builds its own messages, so get_messages() never sees its notes —
    but a reply it leaves in the history ends the turn its next check caches up to."""
    manager, sections = _seed_full_rebuild(prompt_test_environment)
    _change(manager, sections, "car_info", "Driving on Kungsgatan")
    note = manager.message_history.transient_entries[0]
    manager.message_history.history = [{"role": "assistant", "content": "Marcus is on his way.",
                                         "timestamp": note["timestamp"] + 1}]

    _change(manager, sections, "car_info", "Parked on Burggrevegatan")

    assert len(manager.message_history.transient_entries) == 2
    assert manager.message_history.transient_entries[0] is note


@pytest.mark.unit
def test_an_unsent_change_undone_leaves_no_note(prompt_test_environment):
    manager, sections = _seed_full_rebuild(prompt_test_environment)
    received = []
    manager.on_prompt_assembled(lambda msgs: received.append(msgs))
    _change(manager, sections, "car_info", "Driving on Kungsgatan")

    _change(manager, sections, "car_info", "car_info_content")

    assert manager.message_history.transient_entries == []
    assert manager._section_current["car_info"] == "car_info_content"
    assert len(received) == 2, "the withdrawal changes the window, so it is pre-heated too"


@pytest.mark.unit
def test_folding_one_sections_note_leaves_anothers_alone(prompt_test_environment):
    manager, sections = _seed_full_rebuild(prompt_test_environment)
    _change(manager, sections, "car_info", "Driving on Kungsgatan")
    _change(manager, sections, "calendar", "Dentist at 10")
    calendar_note = manager.message_history.transient_entries[1]

    _change(manager, sections, "car_info", "Parked on Burggrevegatan")

    entries = manager.message_history.transient_entries
    assert [e["section"] for e in entries] == ["calendar", "car_info"], "the folded note moves to the end"
    assert entries[0] is calendar_note


@pytest.mark.unit
def test_full_rebuild_resnapshots_baselines_so_next_delta_is_small(prompt_test_environment):
    """After a full rebuild the baselines must reflect the new content — otherwise
    the very next tiny update would diff against stale bytes and trip the
    'large diff' path inappropriately."""
    manager, sections = _seed_full_rebuild(prompt_test_environment)

    # Multi-line content so a one-line change produces a small unified-diff body
    # (single line is fine for the rebuild path but breaks the tiny-delta path
    # because unified_diff prints the whole replaced line on both - and + sides).
    big = "\n".join(f"line{i}" for i in range(PromptManager._DIFF_SIZE_THRESHOLD))
    sections["calendar"] = DummySection(big)
    manager.sections = sections
    manager.on_section_updated("calendar")  # triggers full rebuild (large diff vs initial baseline)

    prompt_after_rebuild = manager.prompt

    # Change a single line — the unified diff is just a couple of lines, well under 200 chars.
    # (Match whole "lineN\n" to avoid hitting "line50"/"line51" as substrings of "line5".)
    new_content = big.replace("line5\n", "line5_changed\n", 1)
    sections["calendar"] = DummySection(new_content)
    manager.sections = sections
    manager.on_section_updated("calendar")  # tiny diff vs the freshly snapshotted baseline

    assert manager.prompt == prompt_after_rebuild, "second update must hit the delta path, not rebuild again"
    assert len(manager.message_history.transient_entries) == 1


@pytest.mark.unit
def test_compute_section_delta_no_change_returns_empty(prompt_test_environment):
    """When old == new there's no diff to send — caller can short-circuit and not
    emit a transient note."""
    assert PromptManager._compute_section_delta("same", "same") == ""


@pytest.mark.unit
def test_compute_section_delta_shows_textual_change(prompt_test_environment):
    diff = PromptManager._compute_section_delta("hello world\nfoo", "hello world\nbar")
    # Minimal diff body: at least one '-foo' and one '+bar' line, no context or hunk headers.
    assert any(line.startswith("-foo") for line in diff.splitlines())
    assert any(line.startswith("+bar") for line in diff.splitlines())
    # No "--- " / "+++ " file headers, no "@@" hunk headers, no unchanged context lines.
    assert "--- " not in diff
    assert "+++ " not in diff
    assert "@@" not in diff
    assert "hello world" not in diff  # context line — must NOT be included


@pytest.mark.unit
def test_compute_section_delta_preserves_content_lines_starting_with_dashes(prompt_test_environment):
    """Regression for an over-broad header filter: content lines beginning with "--" or
    "++" get the diff's "-"/"+" prefix added, producing "---..." / "+++..." in the
    body. A prefix-only filter would incorrectly drop these as if they were headers,
    silently losing real changes."""
    old = "alpha\n--dashed line\ngamma"
    new = "alpha\n++added line\ngamma"
    diff = PromptManager._compute_section_delta(old, new)
    # The removed line was "--dashed line", which appears as "---dashed line" in the diff.
    assert any("dashed line" in line for line in diff.splitlines())
    assert any("added line" in line for line in diff.splitlines())


# ---------------------------------------------------------------------------
# Production notification path: generators identify themselves by display name
# (PromptSectionGenerator.section_name), not by the dict key. The dict lookup
# must translate between them — otherwise the whole feature is a silent no-op
# in production (which was the case before the codex-rescue review caught it).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_on_section_updated_resolves_display_name_to_placeholder_key(prompt_test_environment):
    """Generators call notify_update(self.section_name) with display names like
    "Procedures" or "CalendarData" (see PromptSectionGenerator.__init__), but self.sections
    is keyed by the lowercase placeholder slug. on_section_updated must translate."""
    manager = PromptManager(test_suite=True)
    # Mimic production: dict key is "procedures", generator's section_name is "Procedures".
    procedures = DummySection("v1", section_name="Procedures")
    manager.sections = {
        "procedures": procedures,
        "watchlist": DummySection("p", section_name="WatchList"),
        "home": DummySection("i", section_name="HomeData"),
        "memory": DummySection("m", section_name="Memory"),
        "calendar": DummySection("c", section_name="CalendarData"),
        "integrations": DummySection("s", section_name="Integrations"),
        "whisper": DummySection("w", section_name="Whisper"),
        "car_info": DummySection("car", section_name="TeslaInfo"),
    }
    manager._last_assembly_time = 0
    # Cold start with display name — must trigger full rebuild (no baseline yet).
    manager.on_section_updated("Procedures")
    assert "v1" in manager.prompt
    assert manager._section_baselines["procedures"] == "v1"

    # Now a small change with the display name — must hit the delta path.
    procedures._content = "v1_extra"
    manager.on_section_updated("Procedures")
    assert "v1_extra" not in manager.prompt, "delta path must keep prompt stable"
    assert len(manager.message_history.transient_entries) == 1
    note = manager.message_history.transient_entries[0]
    # Note text carries the display name (what was passed to on_section_updated),
    # not the dict key.
    assert "`Procedures`" in note["content"]


@pytest.mark.unit
def test_diff_threshold_inclusive_boundary(prompt_test_environment, monkeypatch):
    """The diff-size comparison uses `<=`, so a diff of EXACTLY _DIFF_SIZE_THRESHOLD
    chars must still take the delta path. Regression guard against an off-by-one
    that would push borderline cases into needless full rebuilds."""
    manager, sections = _seed_full_rebuild(prompt_test_environment)
    prompt_before = manager.prompt

    threshold = PromptManager._DIFF_SIZE_THRESHOLD
    monkeypatch.setattr(
        PromptManager, "_compute_section_delta",
        staticmethod(lambda old, new: "X" * threshold),
    )

    sections["procedures"] = DummySection("procedures_content_v2")
    manager.sections = sections
    manager.on_section_updated("procedures")

    assert manager.prompt == prompt_before, "diff length == threshold must still hit the delta path"
    assert len(manager.message_history.transient_entries) == 1


@pytest.mark.unit
def test_sections_whose_display_names_share_a_prefix_route_to_their_own_baselines(prompt_test_environment):
    """Regression: two sections once shared a display name (a copy-paste bug). PromptManager
    routes deltas by that display name, so a duplicate would silently mis-attribute one
    section's updates to the other's baseline. Names that merely share a prefix must be
    addressable independently."""
    manager = PromptManager(test_suite=True)
    memory = DummySection("memory_v1", section_name="Memory")
    recent = DummySection("recent_v1", section_name="MemoryRecent")
    manager.sections = {
        "procedures": DummySection("w", section_name="Procedures"),
        "watchlist": DummySection("p", section_name="WatchList"),
        "home": DummySection("i", section_name="HomeData"),
        "memory": memory,
        "memory_recent": recent,
        "calendar": DummySection("c", section_name="CalendarData"),
        "integrations": DummySection("s", section_name="Integrations"),
        "whisper": DummySection("h", section_name="Whisper"),
        "car_info": DummySection("car", section_name="TeslaInfo"),
    }
    manager._last_assembly_time = 0
    # Cold-start full rebuild snapshots BOTH baselines.
    manager.on_section_updated("Memory")
    assert manager._section_baselines["memory"] == "memory_v1"
    assert manager._section_baselines["memory_recent"] == "recent_v1"

    # A memory_recent update must diff against memory_recent's baseline, not memory's.
    recent._content = "recent_v1_extra"
    manager.on_section_updated("MemoryRecent")
    assert len(manager.message_history.transient_entries) == 1
    note = manager.message_history.transient_entries[0]
    assert "`MemoryRecent`" in note["content"]
    # And the memory baseline must still match the unchanged memory content.
    assert manager._section_baselines["memory"] == "memory_v1"


@pytest.mark.unit
def test_assert_unique_section_names_raises_on_duplicates(prompt_test_environment):
    """Defensive guard: if two generators ever share a display name again, the
    duplicate-detection check must raise — silently routing one section's updates
    to the other's baseline would be much harder to diagnose in production."""
    sections = {
        "memory": DummySection("a", section_name="Memory"),
        "memory_recent": DummySection("b", section_name="Memory"),  # collision
        "procedures": DummySection("c", section_name="Procedures"),
    }
    with pytest.raises(RuntimeError, match="Duplicate section_name.*Memory"):
        PromptManager._assert_unique_section_names(sections)


@pytest.mark.unit
def test_diff_threshold_one_over_forces_rebuild(prompt_test_environment, monkeypatch):
    """Just past the threshold must rebuild — proves the boundary check actually
    distinguishes the two paths, not just always-delta or always-rebuild."""
    manager, sections = _seed_full_rebuild(prompt_test_environment)
    received = []
    manager.on_prompt_assembled(lambda msgs: received.append(msgs))

    threshold = PromptManager._DIFF_SIZE_THRESHOLD
    monkeypatch.setattr(
        PromptManager, "_compute_section_delta",
        staticmethod(lambda old, new: "X" * (threshold + 1)),
    )

    sections["procedures"] = DummySection("procedures_content_v2")
    manager.sections = sections
    manager.on_section_updated("procedures")

    assert len(received) == 1, "diff length == threshold + 1 must force a full rebuild"
    assert manager.message_history.transient_entries == []


# ===========================================================================
# Wake-word substitution (commit ae8f065a)
# ===========================================================================
#
# The wake word is configured via `wake_word` in config.yaml and substituted
# into the prompt as {WAKE_WORD}, the same way {PHOTO_COUNT} works. Both the
# live assembly path (_assemble_prompt) and the test-suite path
# (_load_test_suite_prompt) do this substitution. If either ever stops doing
# it, the literal "{WAKE_WORD}" leaks to the LLM — it sees "You are {WAKE_WORD}"
# and the assistant loses its name. Tests are explicit per path because the two
# code paths share no implementation.


def _patch_wake_word(monkeypatch, value, aliases=None):
    """Make config.get('wake_word')/('wake_word_aliases') return chosen values.
    Other keys keep falling through to the real ConfigManager so unrelated
    PromptManager reads don't break."""
    from common.ConfigManager import config as cfg

    real_get = cfg.get

    def fake_get(key, default=None):
        if key == 'wake_word':
            return value if value is not None else default
        if key == 'wake_word_aliases':
            return aliases if aliases is not None else (default if default is not None else [])
        return real_get(key, default)

    monkeypatch.setattr(cfg, 'get', fake_get)


@pytest.mark.unit
def test_testsuite_wake_word_placeholder_substituted_from_config(tmp_path, monkeypatch):
    """The test-suite prompt-load path replaces {WAKE_WORD} with the configured
    name. A regression here would let "You are {WAKE_WORD}" reach the model
    during the prompt test suite."""
    res_dir = tmp_path / "res"
    res_dir.mkdir()
    (res_dir / "prompt.md").write_text("ignored — test_suite path reads testsuite_prompt.md")
    (tmp_path / "testsuite_prompt.md").write_text("You are {WAKE_WORD}, an assistant.")
    monkeypatch.chdir(tmp_path)

    _patch_wake_word(monkeypatch, "Floxy")

    manager = PromptManager(test_suite=True)

    assert manager.prompt == "You are Floxy, an assistant."
    assert "{WAKE_WORD}" not in manager.prompt


@pytest.mark.unit
def test_destinations_placeholder_follows_the_registry(tmp_path, monkeypatch):
    """{DESTINATIONS} is filled from assistant.Destinations, and a registration after the
    manager is up rebuilds the prompt so the new destination is in it."""
    from assistant import Destinations
    Destinations.clear()
    template = (
        "Post to: {DESTINATIONS}\n"
        "{PROCEDURES}\n{WATCHLIST}\n{HOME}\n{MEMORY}\n"
        "{CALENDAR}\n{INTEGRATIONS}\n{WHISPER}\n{CAR_INFO}\n"
    )
    res_dir = tmp_path / "res"
    res_dir.mkdir()
    (res_dir / "prompt.md").write_text(template)
    (tmp_path / "testsuite_prompt.md").write_text(template)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(PromptManager, "_get_photo_count", lambda self: 0)
    try:
        manager = PromptManager(test_suite=True)
        manager.sections = {key: DummySection(f"{key}_content") for key in
                            ["procedures", "watchlist", "home", "memory",
                             "calendar", "integrations", "whisper", "car_info"]}
        manager._assemble_prompt(force=True)
        assert f"Post to: {Destinations.NONE_YET}" in manager.prompt

        Destinations.register(Destinations.Destination("kitchen", Destinations.KIND_VOICE, lambda text: True))
        manager._assemble_prompt(force=True)   # the live manager's listener rebuilds the same way
        assert "Post to: kitchen (speakers)" in manager.prompt
        assert "{DESTINATIONS}" not in manager.prompt
    finally:
        Destinations.clear()


@pytest.mark.unit
def test_live_assembly_wake_word_placeholder_substituted_from_config(tmp_path, monkeypatch):
    """The live _assemble_prompt path replaces {WAKE_WORD} the same way."""
    template = (
        "You are {WAKE_WORD}.\n"
        "{PROCEDURES}\n{WATCHLIST}\n{HOME}\n{MEMORY}\n"
        "{CALENDAR}\n{INTEGRATIONS}\n{WHISPER}\n{CAR_INFO}\n"
    )
    res_dir = tmp_path / "res"
    res_dir.mkdir()
    (res_dir / "prompt.md").write_text(template)
    (tmp_path / "testsuite_prompt.md").write_text(template)
    monkeypatch.chdir(tmp_path)

    _patch_wake_word(monkeypatch, "Aria")

    manager = PromptManager(test_suite=True)
    sections = {key: DummySection(f"{key}_content") for key in
                ["procedures", "watchlist", "home", "memory",
                 "calendar", "integrations", "whisper", "car_info"]}
    manager.sections = sections
    # Drive a real assembly through the section-update path.
    manager.on_section_updated("procedures")

    assert "You are Aria." in manager.prompt
    assert "{WAKE_WORD}" not in manager.prompt


@pytest.mark.unit
def test_wake_word_falls_back_to_michelle_when_unset(tmp_path, monkeypatch):
    """No `wake_word` in config → fall back to "Michelle" so existing deployments
    that haven't added the key keep behaving as they did before the feature."""
    res_dir = tmp_path / "res"
    res_dir.mkdir()
    (res_dir / "prompt.md").write_text("ignored")
    (tmp_path / "testsuite_prompt.md").write_text("You are {WAKE_WORD}.")
    monkeypatch.chdir(tmp_path)

    # value=None routes both real-vs-fake branches to the caller's default
    # ("Michelle"), simulating an absent key.
    _patch_wake_word(monkeypatch, None)

    manager = PromptManager(test_suite=True)

    assert manager.prompt == "You are Michelle."


# ===========================================================================
# {PHOTO_COUNT} tests — fetched once at start, refreshed at most daily, so the
# ever-creeping live count can't flap the cached system-prompt bytes
# ===========================================================================

def _patch_photo_db(monkeypatch, count=12345):
    """Replace FloxyDatabase with a mock whose get_photo_count returns `count`."""
    mock_db = Mock()
    mock_db.get_photo_count.return_value = count
    monkeypatch.setattr("assistant.prompt.PromptManager.FloxyDatabase",
                        Mock(get_instance=Mock(return_value=mock_db)))
    return mock_db


@pytest.mark.unit
def test_photo_count_rendered_approximate_and_fetched_once(prompt_test_environment, monkeypatch):
    """{PHOTO_COUNT} renders as ~N, and rebuilds within the 24h window reuse the
    cached count instead of re-reading the DB — re-fetching on every rebuild is
    what used to change the prompt bytes and invalidate the Anthropic cache."""
    root = prompt_test_environment["root"]
    (root / "res" / "prompt.md").write_text("Photos: {PHOTO_COUNT}\n{PROCEDURES}\n")
    mock_db = _patch_photo_db(monkeypatch, count=12345)

    manager = PromptManager(test_suite=True)
    manager.sections = {"procedures": DummySection("Procedures details")}
    manager._assemble_prompt(force=True)
    assert "Photos: ~12345" in manager.prompt

    mock_db.get_photo_count.return_value = 12399  # count crept up as photos synced
    manager._assemble_prompt(force=True)
    assert mock_db.get_photo_count.call_count == 1
    assert "Photos: ~12345" in manager.prompt  # bytes unchanged within the window


@pytest.mark.unit
def test_photo_count_refreshes_after_daily_window(prompt_test_environment, monkeypatch):
    """Once the 24h cadence elapses, the next read re-fetches from the DB."""
    mock_db = _patch_photo_db(monkeypatch, count=100)
    manager = PromptManager(test_suite=True)
    assert manager._get_photo_count() == 100

    mock_db.get_photo_count.return_value = 250
    manager._photo_count_fetched_at = time.time() - PromptManager._PHOTO_COUNT_REFRESH_S - 1
    assert manager._get_photo_count() == 250
    assert mock_db.get_photo_count.call_count == 2


@pytest.mark.unit
def test_photo_count_failed_refresh_keeps_stale_value(prompt_test_environment, monkeypatch):
    """A failed daily refresh serves the stale count instead of raising or
    flapping the prompt bytes."""
    mock_db = _patch_photo_db(monkeypatch, count=100)
    manager = PromptManager(test_suite=True)
    assert manager._get_photo_count() == 100

    mock_db.get_photo_count.side_effect = RuntimeError("db locked")
    manager._photo_count_fetched_at = time.time() - PromptManager._PHOTO_COUNT_REFRESH_S - 1
    assert manager._get_photo_count() == 100


@pytest.mark.unit
def test_photo_count_first_fetch_failure_raises(prompt_test_environment, monkeypatch):
    """With no cached value to fall back on, a failed first fetch must surface —
    there is no stale count to serve, and baking a broken value into the prompt
    would be worse than failing loudly."""
    mock_db = _patch_photo_db(monkeypatch)
    mock_db.get_photo_count.side_effect = RuntimeError("db locked")
    manager = PromptManager(test_suite=True)
    with pytest.raises(RuntimeError):
        manager._get_photo_count()


# --- toolkits (plans/TOOLKITS_PLAN.md) --------------------------------------------------------

@pytest.mark.unit
def test_toolkit_placeholders_filled_in_both_prompt_paths(tmp_path, monkeypatch):
    from assistant.commands import Toolkits
    res_dir = tmp_path / "res"
    res_dir.mkdir()
    template = "Head\n{TOOLKITS}\n--\n{PROCEDURES}\n"
    (res_dir / "prompt.md").write_text(template)
    (tmp_path / "testsuite_prompt.md").write_text(template)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(PromptManager, "_get_photo_count", lambda self: 0)

    for test_suite in (True, False):
        with patch("assistant.prompt.PromptSectionGenerator.PromptSectionGenerator.start"):
            manager = PromptManager(test_suite=test_suite)
        prompt = manager.prompt
        assert "{TOOLKITS}" not in prompt
        assert Toolkits.render_index() in prompt
        assert "`turn_on(" not in prompt


@pytest.mark.unit
def test_render_template_fills_sections_and_config_values(prompt_test_environment):
    manager = PromptManager(test_suite=True)
    manager.sections = {"procedures": DummySection("sunny")}
    rendered = manager.render_template("a {PROCEDURES} b {WAKE_INTERVAL_MINUTES} c {WAKE_WORD}")
    assert rendered.startswith("a sunny b ")
    assert "{" not in rendered


@pytest.mark.unit
@pytest.mark.parametrize("tts, carries_tags", [({"emotions": True}, True), ({"emotions": False}, False), ({}, False)])
def test_voice_toolkit_carries_the_emotion_tags_only_while_tts_emotions_is_on(prompt_test_environment, monkeypatch,
                                                                              tts, carries_tags):
    from assistant.prompt import PromptManager as module
    from assistant.commands import Toolkits
    real_get = module.config.get
    monkeypatch.setattr(module.config, "get", lambda key, default=None: tts if key == "tts" else real_get(key, default))
    manager = PromptManager(test_suite=True)
    rendered = Toolkits.render(Toolkits.by_name("voice"), manager.render_template)
    assert ("[whispers]" in rendered) is carries_tags
    assert "{EMOTION_TAGS}" not in rendered
    assert "false_positive()" in rendered
    assert "\n\n\n" not in rendered


@pytest.mark.unit
def test_section_without_a_placeholder_emits_no_delta_note(tmp_path, monkeypatch):
    """A section whose data reaches the LLM another way has nothing in the prompt to diff
    against, so a change to it must not produce a 'section updated' note."""
    res_dir = tmp_path / "res"
    res_dir.mkdir()
    (res_dir / "prompt.md").write_text("Head\n{PROCEDURES}\n")
    (tmp_path / "testsuite_prompt.md").write_text("Head\n{PROCEDURES}\n")
    monkeypatch.chdir(tmp_path)
    manager = PromptManager(test_suite=True)
    manager.sections = {"procedures": DummySection("w1"), "phones": DummySection("p1")}
    manager._last_assembly_time = 0
    manager.on_section_updated("procedures")          # cold start: full rebuild, baselines set
    manager.sections["phones"] = DummySection("p2")
    manager.on_section_updated("phones")
    assert manager.message_history.transient_entries == []
    manager.sections["procedures"] = DummySection("w2")
    manager.on_section_updated("procedures")
    assert len(manager.message_history.transient_entries) == 1


# ===========================================================================
# Carried sections: the Indigo data goes with the home toolkit
# ===========================================================================

def _carried_manager(prompt_test_environment, in_window=("home",)):
    """A live-mode manager whose template has no {HOME}: indigo is carried by home."""
    (prompt_test_environment["root"] / "res" / "prompt.md").write_text("Head\n{PROCEDURES}\n")
    manager = PromptManager(test_suite=True)
    manager.test_suite = False
    manager.sections = {"home": DummySection('## LIGHTS\n"Lamp [Kök]": off', section_name="HomeData"),
                        "procedures": DummySection("Sunny", section_name="Procedures")}
    manager._section_positions = manager._compute_section_positions()
    manager.message_history.loaded_toolkits = lambda *_, **__: {name: 1.0 for name in in_window}
    manager._assemble_prompt()
    return manager


def _notes(manager, key="home"):
    return [e["content"] for e in manager.message_history.transient_entries if e["section"] == key]


POINTER = "HOME AUTOMATION DATA has changed since it was sent: run `load_toolkit(home)` for the current state."


@pytest.mark.unit
def test_a_change_before_any_load_with_home_out_of_the_window_is_nobodys_business(prompt_test_environment):
    manager = _carried_manager(prompt_test_environment, in_window=())
    manager.sections["home"]._content = '## LIGHTS\n"Lamp [Kök]": on'
    manager.on_section_updated("HomeData")
    assert _notes(manager) == []


@pytest.mark.unit
def test_carried_data_is_sent_current_and_starts_the_tracking(prompt_test_environment):
    from assistant.commands import Toolkits
    manager = _carried_manager(prompt_test_environment)
    assert "{HOME}" not in manager.prompt and "Lamp" not in manager.prompt
    assert "home" not in manager._section_baselines and "home" not in manager._section_current

    manager.sections["home"]._content = '## LIGHTS\n"Lamp [Kök]": on'
    manager.on_section_updated("HomeData")
    assert _notes(manager) == [POINTER], "home is in the window (from before a restart) with nothing sent since"
    manager.sections["home"]._content = '## LIGHTS\n"Lamp [Kök]": 50'
    manager.on_section_updated("HomeData")
    assert _notes(manager) == [POINTER], "one pointer, not one per change"

    data = manager.carried_data(Toolkits.by_name("home"))
    assert _notes(manager) == [POINTER], "the pointer stays in the window; the send supersedes it"
    assert data == '## HOME AUTOMATION DATA\n## LIGHTS\n"Lamp [Kök]": 50\n'
    assert manager._section_current["home"] == '## LIGHTS\n"Lamp [Kök]": 50'
    assert manager.carried_data(Toolkits.by_name("web")) == ""
    manager.sections["home"]._content = '## LIGHTS\n"Lamp [Kök]": 70'
    manager.on_section_updated("HomeData")
    assert _notes(manager)[-1].endswith('-"Lamp [Kök]": 50\n+"Lamp [Kök]": 70'), "tracking starts from the send"


@pytest.mark.unit
def test_carried_section_changes_become_notes_while_the_toolkit_is_in_the_window(prompt_test_environment):
    from assistant.commands import Toolkits
    manager = _carried_manager(prompt_test_environment)
    manager.carried_data(Toolkits.by_name("home"))
    received = []
    manager.on_prompt_assembled(lambda msgs: received.append(msgs))

    manager.sections["home"]._content = '## LIGHTS\n"Lamp [Kök]": on'
    manager.on_section_updated("HomeData")
    manager.on_section_updated("HomeData")      # unchanged since the note: nothing
    _send(manager)
    manager.sections["home"]._content = '## LIGHTS\n"Lamp [Kök]": 50'
    manager.on_section_updated("HomeData")

    notes = _notes(manager)
    assert len(notes) == 2 and len(received) == 2, "one note and one pre-heat per change"
    assert notes[0] == 'HOME AUTOMATION DATA updated:\n-"Lamp [Kök]": off\n+"Lamp [Kök]": on'
    assert notes[1].endswith('-"Lamp [Kök]": on\n+"Lamp [Kök]": 50'), "each note diffs against the previous one"
    assert "HomeData" not in "\n".join(notes)


@pytest.mark.unit
def test_carried_notes_survive_a_full_rebuild_and_a_new_send_restarts_the_budget(prompt_test_environment, monkeypatch):
    from assistant.commands import Toolkits
    monkeypatch.setattr(PromptManager, "_CARRIED_NOTES_BUDGET", 120)
    manager = _carried_manager(prompt_test_environment)
    manager.carried_data(Toolkits.by_name("home"))
    manager.sections["home"]._content = '## LIGHTS\n"Lamp [Kök]": on'
    manager.on_section_updated("HomeData")
    manager.sections["procedures"]._content = "Rain"
    manager.on_section_updated("Procedures")
    assert len(manager.message_history.transient_entries) == 2

    manager._assemble_prompt()
    assert len(manager.message_history.transient_entries) == 2, "the rebuild changes the prompt, not the window"
    assert manager._section_current["home"] == '## LIGHTS\n"Lamp [Kök]": on', "the last-sent state survives"

    manager.carried_data(Toolkits.by_name("home"))
    assert len(_notes(manager)) == 1, "the old note stays in the window; the fresh data supersedes it"
    for level in (10, 20, 30):
        manager.sections["home"]._content = f'## LIGHTS\n"Lamp [Kök]": {level}'
        manager.on_section_updated("HomeData")
    assert POINTER not in _notes(manager), "the budget counts from the send, not from the notes before it"


@pytest.mark.unit
def test_a_toolkit_preloaded_for_voice_keeps_its_delta_tracking(prompt_test_environment):
    """A home preloaded for a spoken request is a `preload:voice` row: in no text window, yet its
    data was sent. A change must become a delta note there, not end the tracking — which made
    the next spoken request re-send the whole house instead of a note."""
    from assistant.commands import Toolkits
    from assistant.prompt.MessageHistory import MessageHistory
    manager = _carried_manager(prompt_test_environment)
    del manager.message_history.loaded_toolkits      # the real one, over the real rows
    manager.message_history.add_command_result("result: \n# Toolkit: home\nhome text", cap=False,
                                               source=MessageHistory.preload_source("voice"))
    assert "home" not in manager.message_history.loaded_toolkits("text")
    manager.carried_data(Toolkits.by_name("home"))
    manager.sections["home"]._content = '## LIGHTS\n"Lamp [Kök]": on'
    manager.on_section_updated("HomeData")
    assert len(_notes(manager)) == 1 and POINTER not in _notes(manager)
    assert manager._section_current["home"] == '## LIGHTS\n"Lamp [Kök]": on', "the tracking goes on"
    assert not manager.untracked_sections(Toolkits.by_name("home")), "the next preload sends no full copy"


@pytest.mark.unit
def test_carried_section_stops_tracking_when_the_toolkit_leaves_the_window(prompt_test_environment):
    from assistant.commands import Toolkits
    manager = _carried_manager(prompt_test_environment)
    manager.carried_data(Toolkits.by_name("home"))
    manager.sections["home"]._content = '## LIGHTS\n"Lamp [Kök]": on'
    manager.on_section_updated("HomeData")
    assert len(_notes(manager)) == 1

    notes = _notes(manager)
    manager.message_history.loaded_toolkits = lambda *_, **__: {}
    manager.sections["home"]._content = '## LIGHTS\n"Lamp [Kök]": 50'
    manager.on_section_updated("HomeData")
    assert _notes(manager) == notes and "home" not in manager._section_current
    manager.sections["home"]._content = '## LIGHTS\n"Lamp [Kök]": 70'
    manager.on_section_updated("HomeData")
    assert _notes(manager) == notes, "no new note, and the ones sent stay"


@pytest.mark.unit
def test_carried_section_big_change_becomes_one_pointer_note(prompt_test_environment):
    from assistant.commands import Toolkits
    manager = _carried_manager(prompt_test_environment)
    manager.carried_data(Toolkits.by_name("home"))
    manager.sections["home"]._content = '## LIGHTS\n"Lamp [Kök]": on'
    manager.on_section_updated("HomeData")
    _send(manager)
    manager.sections["home"]._content = "## LIGHTS\n" + "\n".join(f'"Lamp {i} [Kök]": on' for i in range(80))
    manager.on_section_updated("HomeData")

    notes = _notes(manager)
    assert len(notes) == 2 and notes[-1] == POINTER, "the pointer follows the note that was sent"
    assert "home" not in manager._section_current
    manager.sections["home"]._content = "## LIGHTS\nnothing"
    manager.on_section_updated("HomeData")
    assert _notes(manager) == notes, "silent until the next load"
    manager._assemble_prompt()
    assert _notes(manager) == notes, "a full rebuild keeps the pointer: no prompt embeds the data"
    assert manager.untracked_sections(Toolkits.by_name("home")) == ["home"]
    manager.carried_data(Toolkits.by_name("home"))
    assert _notes(manager) == notes and manager.untracked_sections(Toolkits.by_name("home")) == []
    manager.sections["home"]._content = "## LIGHTS\nsomething"
    manager.on_section_updated("HomeData")
    assert len(_notes(manager)) == 3 and _notes(manager)[-1] != POINTER, "the send restarts the tracking"


@pytest.mark.unit
def test_carried_notes_budget_ends_in_a_pointer_note(prompt_test_environment, monkeypatch):
    from assistant.commands import Toolkits
    monkeypatch.setattr(PromptManager, "_CARRIED_NOTES_BUDGET", 120)
    manager = _carried_manager(prompt_test_environment)
    manager.carried_data(Toolkits.by_name("home"))
    for level in (10, 20, 30, 40, 50):
        manager.sections["home"]._content = f'## LIGHTS\n"Lamp [Kök]": {level}'
        manager.on_section_updated("HomeData")
        _send(manager)
    notes = _notes(manager)
    assert notes[-1] == POINTER and POINTER not in notes[:-1]
    assert 1 <= len(notes) - 1 < 5, "the notes that fit the budget were sent and stay; the pointer ends the run"


@pytest.mark.unit
def test_carried_notes_budget_counts_only_the_folded_note(prompt_test_environment, monkeypatch):
    """Unsent changes fold into one note: the budget is charged for that note alone, so a
    run of small changes no request saw never ends in a pointer."""
    from assistant.commands import Toolkits
    monkeypatch.setattr(PromptManager, "_CARRIED_NOTES_BUDGET", 120)
    manager = _carried_manager(prompt_test_environment)
    manager.carried_data(Toolkits.by_name("home"))
    for level in (10, 20, 30, 40, 50):
        manager.sections["home"]._content = f'## LIGHTS\n"Lamp [Kök]": {level}'
        manager.on_section_updated("HomeData")

    notes = _notes(manager)
    assert notes == ['HOME AUTOMATION DATA updated:\n-"Lamp [Kök]": off\n+"Lamp [Kök]": 50']
    assert manager._carried_note_chars["home"] == len('-"Lamp [Kök]": off\n+"Lamp [Kök]": 50')


@pytest.mark.unit
def test_add_command_result_passes_the_cap_through(prompt_test_environment):
    """RequestProcessor calls the prompt manager, not the history: the keyword has to
    survive the wrapper (the first console request died on it)."""
    manager = PromptManager(test_suite=True)
    with patch.object(manager.message_history, "add_command_result") as add:
        manager.add_command_result("result: \n# Toolkit: web", cap=False)
        manager.add_command_result("result: \npage")
    assert add.call_args_list == [call("result: \n# Toolkit: web", cap=False, source=None),
                                  call("result: \npage", cap=True, source=None)]


@pytest.mark.unit
def test_after_a_rebuild_the_next_change_diffs_from_the_rebuilt_prompt_and_the_old_note_stays(prompt_test_environment):
    """A rebuild bakes an unsent note's content into the prompt and re-baselines: the next
    change to that section is a note of its own, diffing from the rebuilt state — folding
    into the old note would rewrite a row that is already in the window."""
    manager, sections = _seed_full_rebuild(prompt_test_environment)
    _change(manager, sections, "car_info", "Driving on Kungsgatan")
    first = manager.message_history.transient_entries[0]

    manager._assemble_prompt(force=True)
    assert "Driving on Kungsgatan" in manager.prompt

    _change(manager, sections, "car_info", "Parked on Burggrevegatan")
    entries = manager.message_history.transient_entries
    assert len(entries) == 2 and entries[0] is first
    assert "-Driving on Kungsgatan" in entries[1]["content"] and "+Parked on Burggrevegatan" in entries[1]["content"]
    assert "car_info_content" not in entries[1]["content"], "the baseline is the rebuilt prompt, not the first one"
