"""MemoryFile.seed(): a start without a bank gets the first-run stub, an existing bank is left alone,
and a bank from before the memory/ directory is moved into it."""
import pytest

from assistant import MemoryFile


@pytest.fixture
def stub(tmp_path):
    path = tmp_path / "first_run_memory.md"
    path.write_text("# Long term memory bank placeholder\n")
    return path


@pytest.mark.unit
class TestSeed:
    def test_a_missing_bank_gets_the_stub(self, tmp_path, stub):
        bank = tmp_path / "memory" / "bank.md"
        assert MemoryFile.seed(str(bank), stub, legacy=None) is True
        assert bank.read_text() == stub.read_text()

    def test_an_existing_bank_is_left_alone(self, tmp_path, stub):
        bank = tmp_path / "memory.md"
        bank.write_text("# The Family\n")
        assert MemoryFile.seed(str(bank), stub, legacy=None) is False
        assert bank.read_text() == "# The Family\n"

    def test_no_stub_means_no_bank(self, tmp_path):
        bank = tmp_path / "memory.md"
        assert MemoryFile.seed(str(bank), tmp_path / "absent.md", legacy=None) is False
        assert not bank.exists()

    def test_the_shipped_stub_exists(self):
        assert MemoryFile.FIRST_RUN_STUB.is_file()
        assert "placeholder" in MemoryFile.FIRST_RUN_STUB.read_text()


@pytest.mark.unit
class TestLegacyBank:
    """An install from before memory/ has its bank in the root: the first start moves it."""

    def test_a_legacy_bank_is_moved_into_place_instead_of_the_stub(self, tmp_path, stub):
        legacy = tmp_path / "memory.md"
        legacy.write_text("# The Family\n* Marcus likes tea\n")
        bank = tmp_path / "memory" / "bank.md"
        assert MemoryFile.seed(str(bank), stub, legacy=str(legacy)) is False
        assert bank.read_text() == "# The Family\n* Marcus likes tea\n"
        assert not legacy.exists()

    def test_a_legacy_bank_is_left_alone_when_the_new_bank_exists(self, tmp_path, stub):
        legacy = tmp_path / "memory.md"
        legacy.write_text("old\n")
        bank = tmp_path / "memory" / "bank.md"
        bank.parent.mkdir()
        bank.write_text("new\n")
        assert MemoryFile.seed(str(bank), stub, legacy=str(legacy)) is False
        assert bank.read_text() == "new\n"
        assert legacy.read_text() == "old\n"


@pytest.mark.unit
class TestWriters:
    def test_append_and_write_create_the_memory_directory(self, tmp_path):
        bank = tmp_path / "memory" / "bank.md"
        MemoryFile.append("Charlotte is allergic to cats", str(bank))
        assert "Charlotte is allergic to cats" in bank.read_text()
        other = tmp_path / "elsewhere" / "bank.md"
        MemoryFile.write("# Bank\n", str(other))
        assert other.read_text() == "# Bank\n"
        assert not (other.parent / "bank.md.tmp").exists()


@pytest.mark.unit
class TestSplit:
    """`split` separates what the curator wrote from what `remember` appended since, so the
    prompt can keep the curated part byte-stable between curations."""

    def test_a_note_appended_after_a_write_is_not_part_of_the_curated_text(self, tmp_path):
        bank = str(tmp_path / "bank.md")
        MemoryFile.write("# The Family\n* Charlotte plays the cello\n", bank)
        MemoryFile.append("Frances moved to Lund", bank)
        curated, since = MemoryFile.split(bank)
        assert curated == "# The Family\n* Charlotte plays the cello\n"
        assert since.strip() == "Frances moved to Lund"

    def test_the_curated_text_stays_identical_as_notes_pile_up(self, tmp_path):
        bank = str(tmp_path / "bank.md")
        MemoryFile.write("# The Family\n", bank)
        MemoryFile.append("Ingeborg prefers tea", bank)
        before, _ = MemoryFile.split(bank)
        MemoryFile.append("Frances moved to Lund", bank)
        after, since = MemoryFile.split(bank)
        assert after == before
        assert since.split() == "Ingeborg prefers tea Frances moved to Lund".split()

    def test_the_two_parts_always_add_up_to_the_file(self, tmp_path):
        bank = tmp_path / "bank.md"
        MemoryFile.write("# The Family\n", str(bank))
        MemoryFile.append("Ingeborg prefers tea", str(bank))
        assert "".join(MemoryFile.split(str(bank))) == bank.read_text()

    def test_a_rewrite_absorbs_the_notes(self, tmp_path):
        bank = str(tmp_path / "bank.md")
        MemoryFile.write("# The Family\n", bank)
        MemoryFile.append("Ingeborg prefers tea", bank)
        MemoryFile.write("# The Family\n* Ingeborg prefers tea\n", bank)
        assert MemoryFile.split(bank) == ("# The Family\n* Ingeborg prefers tea\n", "")

    def test_a_bank_never_written_here_counts_as_curated_whole_and_later_notes_as_since(self, tmp_path):
        bank = tmp_path / "bank.md"
        bank.write_text("# The Family\n* Charlotte plays the cello\n")
        assert MemoryFile.split(str(bank)) == ("# The Family\n* Charlotte plays the cello\n", "")
        MemoryFile.append("Frances moved to Lund", str(bank))
        assert MemoryFile.split(str(bank))[1].strip() == "Frances moved to Lund"

    def test_a_hand_edit_of_the_curated_part_makes_the_whole_file_curated(self, tmp_path):
        bank = tmp_path / "bank.md"
        MemoryFile.write("# The Family\n* Charlotte plays the cello\n", str(bank))
        MemoryFile.append("Frances moved to Lund", str(bank))
        bank.write_text(bank.read_text().replace("cello", "viola"))
        curated, since = MemoryFile.split(str(bank))
        assert since == ""
        assert "viola" in curated and "Frances moved to Lund" in curated

    def test_a_missing_bank_splits_into_nothing_and_creates_nothing(self, tmp_path):
        assert MemoryFile.split(str(tmp_path / "bank.md")) == ("", "")
        assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
class TestCuratedLength:
    """`curated_length` is `split` for a reader that only looks — the console's Bank tab: it
    says where the curated part ends and never marks anything."""

    def test_it_is_where_the_notes_start(self, tmp_path):
        bank = str(tmp_path / "bank.md")
        MemoryFile.write("# The Family\n", bank)
        MemoryFile.append("Ingeborg prefers tea", bank)
        text = MemoryFile.read(bank)
        offset = MemoryFile.curated_length(bank)
        assert (text[:offset], text[offset:]) == MemoryFile.split(bank)

    def test_a_bank_never_written_here_has_none_and_gets_no_mark(self, tmp_path):
        bank = tmp_path / "bank.md"
        bank.write_text("# The Family\n* Charlotte plays the cello\n")
        assert MemoryFile.curated_length(str(bank)) is None
        assert [p.name for p in tmp_path.iterdir()] == ["bank.md"]

    def test_a_hand_edit_of_the_curated_part_has_none_and_leaves_the_mark_alone(self, tmp_path):
        bank = tmp_path / "bank.md"
        MemoryFile.write("# The Family\n* Charlotte plays the cello\n", str(bank))
        mark = (tmp_path / "bank.md.curated").read_text()
        bank.write_text("# The Family\n* Charlotte plays the violin\n")
        assert MemoryFile.curated_length(str(bank)) is None
        assert (tmp_path / "bank.md.curated").read_text() == mark

    def test_a_bank_cut_shorter_than_its_mark_has_none(self, tmp_path):
        bank = tmp_path / "bank.md"
        MemoryFile.write("# The Family\n* Charlotte plays the cello\n", str(bank))
        bank.write_text("# The Family\n")
        assert MemoryFile.curated_length(str(bank)) is None

    def test_a_missing_bank_has_none_and_creates_nothing(self, tmp_path):
        assert MemoryFile.curated_length(str(tmp_path / "bank.md")) is None
        assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
class TestNoThinkTags:
    """A think tag in the bank ends a local model's thinking when it is quoted there."""

    NOTE = "* The leak: a bare `</think>` outside a block, then `<think>` again. Also </ Think > and <THINKING>."

    @staticmethod
    def assert_clean(text):
        assert not MemoryFile.THINK_TAG.search(text)
        assert "think>" not in text.lower().replace(" ", "")
        assert text.startswith("* The leak: a bare `") and "outside a block" in text

    def test_a_written_bank_holds_no_think_tag(self, tmp_path):
        bank = str(tmp_path / "bank.md")
        MemoryFile.write(self.NOTE, bank)
        self.assert_clean(MemoryFile.read(bank))

    def test_a_remembered_note_holds_no_think_tag(self, tmp_path):
        bank = str(tmp_path / "bank.md")
        MemoryFile.append(self.NOTE, bank)
        self.assert_clean(MemoryFile.read(bank).strip())

    def test_the_cleaned_bank_still_counts_as_curated_whole(self, tmp_path):
        bank = str(tmp_path / "bank.md")
        MemoryFile.write(self.NOTE, bank)
        curated, since = MemoryFile.split(bank)
        assert since == "" and curated == MemoryFile.read(bank)

    def test_the_opening_and_the_closing_tag_stay_told_apart(self):
        opening, closing = MemoryFile.without_think_tags("<think>"), MemoryFile.without_think_tags("</think>")
        assert opening != closing

    def test_other_tags_and_plain_words_are_left_alone(self):
        text = "The <memory> block; I think > you; <b>bold</b>; rethink it."
        assert MemoryFile.without_think_tags(text) == text
