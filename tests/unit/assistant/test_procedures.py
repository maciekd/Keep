"""Procedures: one markdown file per how-to; the index is built from the files, a replaced or
deleted version lands in the log directory, and the curator's mode can add but never replace."""
import pytest

from assistant import Procedures

BODY = '1) open shop.example/cart\n2) click "Pay", never press Back\n`fetch(\'/pay?id=\' + id)`'


@pytest.fixture
def pd(tmp_path):
    return tmp_path / "procedures"


@pytest.fixture
def log(tmp_path):
    return tmp_path / "log"


def log_texts(log) -> list[str]:
    return [p.read_text() for p in sorted(log.glob("*.md"))] if log.is_dir() else []


@pytest.mark.unit
class TestSave:
    def test_save_creates_a_file_that_reads_back_with_the_text_verbatim(self, pd, log):
        assert Procedures.save("Shop checkout", "paying at shop.example", BODY, pd, log) == ""
        text = Procedures.read("shop-checkout", pd)
        assert BODY in text
        assert Procedures.when(text) == "paying at shop.example"
        assert Procedures.names(pd) == ["shop-checkout"]
        assert log_texts(log) == []

    @pytest.mark.parametrize("name", ["Shop Checkout", "shop-checkout", '"shop checkout"', " SHOP_CHECKOUT "])
    def test_a_procedure_answers_to_every_spelling_of_its_name(self, pd, log, name):
        Procedures.save("shop-checkout", "paying", BODY, pd, log)
        assert Procedures.read(name, pd) is not None

    def test_replacing_copies_the_previous_version_to_the_log(self, pd, log):
        Procedures.save("shop-checkout", "paying", "old steps", pd, log)
        assert Procedures.save("Shop Checkout", "paying", "new steps", pd, log) == ""
        assert "new steps" in Procedures.read("shop-checkout", pd)
        assert Procedures.names(pd) == ["shop-checkout"]
        texts = log_texts(log)
        assert len(texts) == 1 and "old steps" in texts[0]

    def test_a_nameless_or_empty_procedure_is_refused(self, pd, log):
        assert Procedures.save("???", "when", BODY, pd, log)
        assert Procedures.save("shop", "when", "   ", pd, log)
        assert Procedures.names(pd) == []

    def test_an_oversized_procedure_is_refused_and_nothing_is_written(self, pd, log):
        failure = Procedures.save("shop", "when", "x" * (Procedures.MAX_FILE_CHARS + 1), pd, log)
        assert failure
        assert Procedures.read("shop", pd) is None

    def test_a_multi_line_when_becomes_one_index_line(self, pd, log):
        Procedures.save("shop", "paying\n  at shop.example", BODY, pd, log)
        assert Procedures.when(Procedures.read("shop", pd)) == "paying at shop.example"


@pytest.mark.unit
class TestWhenLimit:
    """A `when` is an index line of every prompt: a save with one over the limit is refused and
    told the limit, and a longer line an older file holds is cut in the index alone."""

    def test_a_when_at_the_limit_is_saved_and_one_character_more_is_refused(self, pd, log):
        fits = "x" * Procedures.WHEN_CHARS
        assert Procedures.save("fits", fits, BODY, pd, log) == ""
        failure = Procedures.run("save_procedure", ["over", fits + "y", BODY], pd, log)
        assert str(Procedures.WHEN_CHARS) in failure and str(Procedures.WHEN_CHARS + 1) in failure
        assert Procedures.names(pd) == ["fits"]

    def test_the_limit_counts_the_line_as_it_is_stored(self, pd, log):
        spaced = "paying   at\n" + " " * Procedures.WHEN_CHARS + "shop.example"
        assert Procedures.save("shop", spaced, BODY, pd, log) == ""

    def test_a_refused_replacement_leaves_the_procedure_as_it_was(self, pd, log):
        Procedures.save("shop", "paying", BODY, pd, log)
        assert Procedures.save("shop", "z" * (Procedures.WHEN_CHARS + 1), "new steps", pd, log)
        assert BODY in Procedures.read("shop", pd) and log_texts(log) == []

    def test_a_longer_line_of_an_older_file_is_cut_at_a_word_in_the_index_and_loads_whole(self, pd, log):
        Procedures.save("shop", "paying", BODY, pd, log)
        long_when = "paying by card " + "at shop.example " * 6 + "LASTWORD"
        path = pd / "shop.md"
        path.write_text(path.read_text().replace("When: paying", f"When: {long_when}"))
        line = Procedures.render_index(pd)
        shown = line.split(" — ", 1)[1]
        assert shown.endswith("…") and len(shown) <= Procedures.WHEN_CHARS + 1
        assert long_when.startswith(shown[:-1]) and long_when[len(shown) - 1] == " "
        assert long_when in Procedures.load("shop", pd)

    def test_the_prompt_states_the_limit_the_save_enforces(self):
        from common.Paths import RES_DIR
        text = (RES_DIR / "prompt.md").read_text(encoding="utf-8")
        declaration = " ".join(text.split("`save_procedure(name", 1)[1].split("`delete_procedure(", 1)[0].split())
        assert f"{Procedures.WHEN_CHARS} characters" in declaration


@pytest.mark.unit
class TestNoReplaceMode:
    """What the memory curator gets: it may add, never replace or delete."""

    def test_a_new_name_is_allowed(self, pd, log):
        assert Procedures.save("shop", "when", BODY, pd, log, allow_replace=False) == ""
        assert Procedures.read("shop", pd) is not None

    def test_replacing_is_refused_and_the_procedure_is_untouched(self, pd, log):
        Procedures.save("shop", "when", "the original steps", pd, log)
        before = Procedures.read("shop", pd)
        assert Procedures.save("shop", "when", "a rewrite", pd, log, allow_replace=False)
        assert Procedures.read("shop", pd) == before
        assert log_texts(log) == []

    def test_run_refuses_delete_but_allows_a_note(self, pd, log):
        Procedures.save("shop", "when", BODY, pd, log)
        assert Procedures.run("delete_procedure", ["shop"], pd, log, allow_replace=False)
        assert Procedures.read("shop", pd) is not None
        assert Procedures.run("procedure_note", ["shop", "3DS may need a retry"], pd, log, allow_replace=False) == ""
        assert "3DS may need a retry" in Procedures.read("shop", pd)


@pytest.mark.unit
class TestNote:
    def test_notes_collect_under_one_heading_and_leave_the_steps_alone(self, pd, log):
        Procedures.save("shop", "when", BODY, pd, log)
        assert Procedures.note("shop", "the push may need a retry", pd) == ""
        assert Procedures.note("Shop", "guest checkout is fine", pd) == ""
        text = Procedures.read("shop", pd)
        assert text.count(Procedures.NOTES_HEADING) == 1
        assert BODY in text
        notes = text.split(Procedures.NOTES_HEADING, 1)[1]
        assert notes.index("the push may need a retry") < notes.index("guest checkout is fine")

    def test_a_note_for_an_unknown_procedure_names_what_exists(self, pd, log):
        Procedures.save("shop", "when", BODY, pd, log)
        failure = Procedures.note("bank-login", "x", pd)
        assert "bank-login" in failure and "shop" in failure

    def test_an_empty_note_is_refused(self, pd, log):
        Procedures.save("shop", "when", BODY, pd, log)
        before = Procedures.read("shop", pd)
        assert Procedures.note("shop", "  ", pd)
        assert Procedures.read("shop", pd) == before

    def test_a_note_that_would_overflow_the_file_is_refused(self, pd, log):
        Procedures.save("shop", "when", "x" * (Procedures.MAX_FILE_CHARS - 200), pd, log)
        before = Procedures.read("shop", pd)
        assert Procedures.note("shop", "y" * 300, pd)
        assert Procedures.read("shop", pd) == before


@pytest.mark.unit
class TestDelete:
    def test_delete_removes_the_file_and_keeps_the_last_version_in_the_log(self, pd, log):
        Procedures.save("shop", "when", BODY, pd, log)
        assert Procedures.delete("Shop", pd, log) == ""
        assert Procedures.names(pd) == []
        assert len(log_texts(log)) == 1 and BODY in log_texts(log)[0]

    def test_deleting_an_unknown_procedure_is_an_error(self, pd, log):
        assert Procedures.delete("shop", pd, log)
        assert log_texts(log) == []


@pytest.mark.unit
class TestIndexAndLoad:
    def test_no_procedures_says_so(self, pd):
        assert Procedures.render_index(pd) == Procedures.EMPTY

    def test_the_index_is_one_line_per_file_from_its_when_line(self, pd, log):
        Procedures.save("School portal login", "anything asked about the school portal", "steps", pd, log)
        Procedures.save("shop-checkout", "paying at shop.example", "steps", pd, log)
        lines = Procedures.render_index(pd).splitlines()
        assert len(lines) == 2
        assert "school-portal-login" in lines[0] and "anything asked about the school portal" in lines[0]
        assert "shop-checkout" in lines[1] and "paying at shop.example" in lines[1]

    def test_the_index_follows_a_hand_edited_file(self, pd, log):
        Procedures.save("shop", "paying", "steps", pd, log)
        path = pd / "shop.md"
        path.write_text(path.read_text().replace("When: paying", "When: paying by card at shop.example"))
        assert "paying by card at shop.example" in Procedures.render_index(pd)

    def test_load_returns_the_procedure_under_a_header_naming_it(self, pd, log):
        Procedures.save("Shop checkout", "paying", BODY, pd, log)
        loaded = Procedures.load("shop checkout", pd)
        assert loaded.startswith(Procedures.HEADER_PREFIX + "shop-checkout")
        assert BODY in loaded

    def test_load_of_an_unknown_name_lists_the_available_ones(self, pd, log):
        Procedures.save("shop-checkout", "paying", BODY, pd, log)
        Procedures.save("school-portal-login", "school", BODY, pd, log)
        answer = Procedures.load("bank", pd)
        assert "shop-checkout" in answer and "school-portal-login" in answer
        assert BODY not in answer

    def test_version_moves_on_save_and_delete(self, pd, log):
        empty = Procedures.version(pd)
        Procedures.save("shop", "when", BODY, pd, log)
        saved = Procedures.version(pd)
        Procedures.delete("shop", pd, log)
        assert empty != saved and Procedures.version(pd) != saved


@pytest.mark.unit
class TestRun:
    def test_save_joins_a_text_that_arrived_split_on_its_commas(self, pd, log):
        assert Procedures.run("save_procedure", ["shop", "paying", "open the cart", "click Pay", "wait"], pd, log) == ""
        assert "open the cart, click Pay, wait" in Procedures.read("shop", pd)

    def test_note_joins_a_split_text_too(self, pd, log):
        Procedures.save("shop", "when", BODY, pd, log)
        assert Procedures.run("procedure_note", ["shop", "retry once", "then give up"], pd, log) == ""
        assert "retry once, then give up" in Procedures.read("shop", pd)

    @pytest.mark.parametrize("command, args", [
        ("save_procedure", ["shop", "when"]),
        ("procedure_note", ["shop"]),
        ("delete_procedure", []),
        ("watch", ["a", "b"]),
    ])
    def test_too_few_arguments_or_a_foreign_command_is_an_error(self, pd, log, command, args):
        assert Procedures.run(command, args, pd, log)
        assert Procedures.names(pd) == []

    def test_delete_through_run(self, pd, log):
        Procedures.save("shop", "when", BODY, pd, log)
        assert Procedures.run("delete_procedure", ["shop"], pd, log) == ""
        assert Procedures.names(pd) == []
