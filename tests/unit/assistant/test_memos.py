"""Memos: lookup material outside the bank. The store shares its mechanics with the procedures
(`MemoryDocs`, covered through `Procedures` in test_procedures.py); here is what a memo is on
its own — its commands, its place in the prompt, and the curator that may replace and delete it."""
from unittest.mock import Mock, patch

import pytest

from assistant import MemoryFile
from assistant.Memos import Memos
from assistant.Procedures import Procedures
from assistant.agents.LongTermMemoryCurator import LongTermMemoryCurator
from assistant.commands.CommandParser import CommandParser
from assistant.commands.Commands import Commands, is_reference_result
from assistant.prompt.sections.MemosSectionGenerator import MemosSectionGenerator

WINES = "Reds: Barolo, Rioja Reserva\nWhites: Chablis\nNever: anything oaked for Charlotte"


@pytest.fixture(autouse=True)
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(Memos, "DIR", tmp_path / "memos")
    monkeypatch.setattr(Memos, "LOG_DIR", tmp_path / "log" / "memos")
    monkeypatch.setattr(Procedures, "DIR", tmp_path / "procedures")
    monkeypatch.setattr(Procedures, "LOG_DIR", tmp_path / "log" / "procedures")
    return tmp_path


def log_texts(tmp_path) -> list[str]:
    return [p.read_text() for p in sorted((tmp_path / "log" / "memos").glob("*.md"))]


@pytest.mark.unit
class TestStore:
    def test_a_memo_is_saved_replaced_and_deleted_with_each_old_version_in_its_own_log(self, stores):
        assert Memos.save("Wine list", "choosing or buying wine", WINES) == ""
        assert Memos.save("wine-list", "choosing or buying wine", WINES + "\nRosé: none") == ""
        assert "Rosé: none" in Memos.read("wine list")
        assert Memos.delete("wine-list") == ""
        assert Memos.names() == []
        old = log_texts(stores)
        assert len(old) == 2 and "Rosé" not in old[0] and "Rosé" in old[1]

    def test_memos_and_procedures_do_not_see_each_other(self):
        Memos.save("wine-list", "choosing wine", WINES)
        Procedures.save("shop-checkout", "paying", "open the cart")
        assert Memos.names() == ["wine-list"] and Procedures.names() == ["shop-checkout"]
        assert "wine-list" in Memos.load("shop-checkout") and WINES not in Procedures.load("wine-list")

    def test_a_loaded_memo_is_told_from_a_loaded_procedure_by_its_header(self):
        Memos.save("wine-list", "choosing wine", WINES)
        Procedures.save("wine-list", "ordering wine", "open the shop")
        memo, procedure = Memos.load("wine-list"), Procedures.load("wine-list")
        assert memo.splitlines()[0] != procedure.splitlines()[0]
        assert is_reference_result(memo) and is_reference_result(procedure)

    def test_a_refusal_names_the_memo_command(self):
        for failure in (Memos.save("wine-list", "w" * (Memos.WHEN_CHARS + 1), WINES),
                        Memos.save("wine-list", "choosing wine", "x" * Memos.MAX_FILE_CHARS),
                        Memos.run("save_memo", ["wine-list"])):
            assert failure.startswith("save_memo failed")
        assert Memos.run("delete_memo", ["cellar"]).startswith("delete_memo failed")
        assert Memos.names() == []

    @pytest.mark.parametrize("command", ["procedure_note", "save_procedure", "delete_procedure"])
    def test_a_procedure_command_is_none_of_the_memos(self, command):
        Memos.save("wine-list", "choosing wine", WINES)
        assert Memos.run(command, ["wine-list", "choosing wine", "gone"])
        assert WINES in Memos.read("wine-list")

    def test_a_memo_command_is_none_of_the_procedures(self):
        assert Procedures.run("save_memo", ["wine-list", "choosing wine", WINES])
        assert Procedures.names() == [] and Memos.names() == []


@pytest.fixture
def commands():
    return Commands(Mock(command_reply=None, command_reply_contains_error=False))


def run(commands: Commands, text: str):
    (name, params), _ = CommandParser(commands.list_commands()).extract_command_and_params(text)
    return commands.run(name, params)


@pytest.mark.unit
class TestCommands:
    def test_save_replace_and_delete_through_the_dispatcher(self, commands):
        assert not run(commands, f'save_memo("Wine list", "choosing wine", """{WINES}""")')
        assert not run(commands, 'save_memo("wine-list", "choosing wine", """Reds only""")')
        assert "Reds only" in Memos.read("wine-list") and "Chablis" not in Memos.read("wine-list")
        assert not run(commands, 'delete_memo("wine-list")')
        assert Memos.names() == []

    def test_store_refusals_come_back_as_the_commands_error(self, commands):
        assert run(commands, 'save_memo("wine-list", "choosing wine")')
        assert "cellar" in run(commands, 'delete_memo("cellar")')

    def test_loading_returns_each_named_memo_and_says_what_exists_otherwise(self, commands):
        Memos.save("wine-list", "choosing wine", WINES)
        Memos.save("ski-trip", "the trip in February", "Hotel Example, booking KEP47")
        reply = commands._run_load_memo("wine list", "ski-trip")
        assert WINES in reply and "KEP47" in reply
        assert "ski-trip" in commands._run_load_memo() and "ski-trip" in commands._run_load_memo("cellar")

    def test_loading_is_a_background_command_that_the_model_waits_for(self, commands):
        from assistant.commands import Toolkits
        commands.background_command = Mock()
        run(commands, 'load_memo("wine-list")')
        function, params = commands.background_command.call_args.args[:2]
        assert params == ["wine-list"] and callable(function)
        assert "load_memo" in Toolkits.LOADS

    def test_loading_runs_in_test_suite_mode_but_writing_does_not(self, commands):
        assert commands.is_pure("load_memo")
        assert not commands.is_pure("save_memo") and not commands.is_pure("delete_memo")

    def test_the_chat_echo_names_the_memo_and_not_its_text(self, commands):
        text = commands.describe("save_memo", ["wine-list", "choosing wine", "a long body, with commas"])
        assert "wine-list" in text and "a long body" not in text


@pytest.mark.unit
class TestPromptSection:
    def test_the_section_is_the_index_never_the_texts(self):
        section = MemosSectionGenerator(None)
        assert section._fetch_content() == Memos.EMPTY
        Memos.save("wine-list", "choosing or buying wine", WINES)
        content = section._fetch_content()
        assert "wine-list" in content and "choosing or buying wine" in content and "Barolo" not in content

    def test_the_loop_refreshes_when_a_memo_changed(self):
        section = MemosSectionGenerator(None)
        Memos.save("wine-list", "choosing wine", WINES)
        with patch.object(section, "_wait", side_effect=StopIteration), pytest.raises(StopIteration):
            section._start_monitoring()
        assert "wine-list" in section.content

    def test_the_prompt_template_has_a_place_for_the_index_and_states_the_when_limit(self):
        """Contract with res/prompt.md: PromptManager fills a section where its placeholder stands."""
        from assistant.prompt.PromptManager import PromptManager
        from common.Paths import RES_DIR
        template = (RES_DIR / "prompt.md").read_text(encoding="utf-8")
        assert "{MEMOS}" in template and "memos" in PromptManager.SECTION_ORDER
        declaration = " ".join(template.split("`save_memo(name", 1)[1].split("`delete_memo(", 1)[0].split())
        assert f"{Memos.WHEN_CHARS} characters" in declaration


@pytest.fixture
def curator(tmp_path):
    llm = Mock(last_stop_reason=None)
    return LongTermMemoryCurator(llm=llm, test_mode=True, memory_file_path=str(tmp_path / "live" / "bank.md"),
                                 watchlist_dir=tmp_path / "live" / "watchlist",
                                 procedures_dir=tmp_path / "live" / "procedures", memos_dir=tmp_path / "live" / "memos")


@pytest.mark.unit
class TestCurator:
    def test_it_saves_replaces_and_deletes_memos_in_its_own_directory(self, curator):
        results, unsaved = curator.run_commands(
            f'save_memo("Wine list", "choosing wine", """{WINES}""")\n'
            'save_memo("ski-trip", "the trip in February", """Hotel Example, booking KEP47""")\n'
            'save_memo("wine-list", "choosing wine", """Reds only""")\n'
            'delete_memo("ski-trip")')
        assert unsaved == [] and not any(r.startswith("FAILED") for r in results)
        assert Memos.names(curator.memos_dir) == ["wine-list"]
        assert "Reds only" in Memos.read("wine-list", curator.memos_dir)
        assert Memos.names() == []

    def test_a_procedure_is_still_not_its_to_replace(self, curator):
        Procedures.save("wine-list", "ordering wine", "the original steps", curator.procedures_dir)
        results, _ = curator.run_commands('save_procedure("wine-list", "ordering wine", """condensed""")')
        assert results[0].startswith("FAILED")
        assert "the original steps" in Procedures.read("wine-list", curator.procedures_dir)

    def test_the_prompt_carries_every_memo_in_full(self, curator):
        Memos.save("wine-list", "choosing wine", WINES, curator.memos_dir)
        curator.llm.think.side_effect = ["<memory># Bank</memory>"]
        curator.save_debug_log = Mock()
        curator.curate_memory_with_data("history", "", "# Bank")
        prompt = curator.llm.think.call_args.kwargs['system_message']
        assert WINES in prompt and "wine-list" in prompt
        assert "save_memo(" in prompt and "delete_memo(" in prompt
        assert "{MEMO" not in prompt

    def test_a_refused_memo_keeps_its_text_in_the_bank(self, curator):
        curator.curate_memory = Mock(return_value=("# Bank\n* Wine: see the wine-list memo\n", {
            'interaction_count': 2, 'new_memory_size': 7,
            'commands': f'save_memo("wine-list", "{"w" * (Memos.WHEN_CHARS + 1)}", """{WINES}""")'}))
        curator.save_debug_log = Mock()
        LongTermMemoryCurator.run(curator=curator)
        bank = MemoryFile.read(curator.memory_file_path)
        assert bank.startswith("# Bank\n* Wine: see the wine-list memo\n") and WINES in bank
        assert Memos.names(curator.memos_dir) == []

    def test_a_memo_the_run_did_not_name_is_not_rewritten(self, curator):
        Memos.save("wine-list", "choosing wine", WINES, curator.memos_dir)
        before = Memos.version(curator.memos_dir)
        curator.run_commands('save_memo("ski-trip", "the trip in February", """Hotel Example""")')
        assert [v for v in Memos.version(curator.memos_dir) if v[0] == "wine-list.md"] == list(before)


@pytest.mark.unit
class TestMemorySizeTelemetry:
    """The nightly memory event counts the memos and their words: the curator's prompt carries
    every one of them in full, so their total is what a night costs."""

    @staticmethod
    def reported() -> dict:
        from assistant.agents.AutonomousActionManager import AutonomousActionManager
        with patch.object(MemoryFile, "read", return_value="Marcus lives with Charlotte."), \
                patch("assistant.agents.AutonomousActionManager.WatchList.items", return_value=[]), \
                patch("assistant.agents.AutonomousActionManager.Telemetry.get_instance") as telemetry:
            AutonomousActionManager._report_memory_size()
        words, chars, watch_items, procedures, memos, memo_words = telemetry.return_value.memory_size.call_args.args
        return {"words": words, "procedures": procedures, "memos": memos, "memo_words": memo_words}

    def test_an_install_without_memos_reports_none(self):
        assert self.reported() == {"words": 4, "procedures": 0, "memos": 0, "memo_words": 0}

    def test_every_memo_is_counted_with_its_words_and_no_procedure_among_them(self):
        assert not Memos.run("save_memo", ["wines", "choosing a bottle", WINES])
        one = self.reported()
        assert one["memos"] == 1 and one["memo_words"] >= len(WINES.split())
        assert not Memos.run("save_memo", ["plumbers", "a leak", "Dave fixes pipes on weekdays only"])
        assert not Procedures.run("save_procedure", ["descale", "the kettle is slow", "Vinegar, then boil twice."])
        two = self.reported()
        assert two["memos"] == 2 and two["procedures"] == 1
        assert two["memo_words"] - one["memo_words"] >= 6, "the second memo's six words of text are in the total"
        assert two["words"] == one["words"], "the bank's own count never includes a memo"
