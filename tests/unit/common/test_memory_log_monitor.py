from datetime import date
from unittest.mock import Mock, patch

import pytest

from common import MemorySizes
from common.MemoryLogMonitor import MemoryLogMonitor


def _make_monitor():
    tab = Mock()
    tab.memory_log_sizes = []
    tab.memory_log_error = ""
    tab.memory_word_count = 0
    return MemoryLogMonitor(tab, fetch_interval=0), tab


@pytest.fixture(autouse=True)
def _no_real_memory_bank(tmp_path):
    with patch.object(MemoryLogMonitor, "MEMORY_FILE", tmp_path / "bank" / "memory.md"):
        yield


def _snapshot_body(new_size_chars: int, extra_bytes: int = 0, folders: dict = None) -> str:
    body = (
        "# Memory Debug Log\n\n"
        "- Timestamp: 2026-01-01 04:00:00\n"
        "- Interactions: 50\n"
        "- Calendar events: 5\n"
        "- Old size: 4000 chars\n"
        f"- New size: {new_size_chars} chars\n"
        + "".join(f"- {MemorySizes.FOLDERS[part]}: {chars} chars\n" for part, chars in (folders or {}).items()) +
        "- LLM duration: 10.0s\n\n"
        "---\n\n"
    )
    if extra_bytes > 0:
        body += "x" * extra_bytes
    return body


def _write(dir_path, name, content):
    p = dir_path / name
    p.write_text(content)
    return p


@pytest.mark.unit
class TestMemoryLogMonitorScan:

    def test_parses_new_size_header_not_file_size(self, tmp_path):
        # File is huge but header says memory is small — must report header value
        _write(tmp_path, "20260101_040000_abc123_memory.md",
               _snapshot_body(new_size_chars=5120, extra_bytes=200_000))

        monitor, tab = _make_monitor()
        with patch.object(MemoryLogMonitor, "LOG_DIR", tmp_path):
            monitor._scan()

        assert len(tab.memory_log_sizes) == 1
        day, sizes = tab.memory_log_sizes[0]
        assert day == date(2026, 1, 1)
        assert sizes == {"bank": 5.0}  # 5120 / 1024

    def test_parses_multiple_files_and_sorts_by_date(self, tmp_path):
        _write(tmp_path, "20260101_040000_abc123_memory.md", _snapshot_body(1024))
        _write(tmp_path, "20260103_040000_def456_memory.md", _snapshot_body(2048))
        _write(tmp_path, "20260102_040000_aaa999_memory.md", _snapshot_body(3072))

        monitor, tab = _make_monitor()
        with patch.object(MemoryLogMonitor, "LOG_DIR", tmp_path):
            monitor._scan()

        assert [d for d, _ in tab.memory_log_sizes] == [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)]
        assert [sizes["bank"] for _, sizes in tab.memory_log_sizes] == [1.0, 3.0, 2.0]
        assert tab.memory_log_error == ""

    def test_missing_directory_sets_error_and_empties_data(self, tmp_path):
        monitor, tab = _make_monitor()
        tab.memory_log_sizes = [(date(2026, 1, 1), 5.0)]
        missing = tmp_path / "does_not_exist"
        with patch.object(MemoryLogMonitor, "LOG_DIR", missing):
            monitor._scan()

        assert tab.memory_log_sizes == []
        assert "not found" in tab.memory_log_error.lower()

    def test_skips_malformed_filenames(self, tmp_path):
        _write(tmp_path, "20260101_040000_abc_memory.md", _snapshot_body(1024))
        _write(tmp_path, "not_a_memory_file.txt", "junk")
        _write(tmp_path, "memory.md", "junk")
        _write(tmp_path, "99999999_040000_abc_memory.md", _snapshot_body(1024))  # bad date

        monitor, tab = _make_monitor()
        with patch.object(MemoryLogMonitor, "LOG_DIR", tmp_path):
            monitor._scan()

        assert len(tab.memory_log_sizes) == 1
        assert tab.memory_log_sizes[0][0] == date(2026, 1, 1)

    def test_skips_files_without_new_size_header(self, tmp_path):
        _write(tmp_path, "20260101_040000_abc123_memory.md",
               "# Memory Debug Log\n\nno header here, just text\n")
        _write(tmp_path, "20260102_040000_def456_memory.md", _snapshot_body(2048))

        monitor, tab = _make_monitor()
        with patch.object(MemoryLogMonitor, "LOG_DIR", tmp_path):
            monitor._scan()

        assert len(tab.memory_log_sizes) == 1
        assert tab.memory_log_sizes[0][0] == date(2026, 1, 2)

    def test_multiple_files_same_day_keeps_largest_memory(self, tmp_path):
        _write(tmp_path, "20260101_040000_aaaaaa_memory.md", _snapshot_body(1024))
        _write(tmp_path, "20260101_120000_bbbbbb_memory.md", _snapshot_body(5120))
        _write(tmp_path, "20260101_180000_cccccc_memory.md", _snapshot_body(3072))

        monitor, tab = _make_monitor()
        with patch.object(MemoryLogMonitor, "LOG_DIR", tmp_path):
            monitor._scan()

        assert len(tab.memory_log_sizes) == 1
        assert tab.memory_log_sizes[0][0] == date(2026, 1, 1)
        assert tab.memory_log_sizes[0][1] == {"bank": 5.0}

    def test_a_snapshot_with_the_folders_sizes_gives_every_part_and_an_older_one_the_bank_alone(self, tmp_path):
        _write(tmp_path, "20260101_040000_aaaaaa_memory.md", _snapshot_body(1024))
        _write(tmp_path, "20260102_040000_bbbbbb_memory.md",
               _snapshot_body(2048, folders={"procedures": 512, "memos": 4096, "watchlist": 3072}))

        monitor, tab = _make_monitor()
        with patch.object(MemoryLogMonitor, "LOG_DIR", tmp_path):
            monitor._scan()

        assert [sizes for _, sizes in tab.memory_log_sizes] == [
            {"bank": 1.0}, {"bank": 2.0, "procedures": 0.5, "memos": 4.0, "watchlist": 3.0}]

    def test_the_header_the_curator_writes_is_the_one_read_back(self, tmp_path):
        from assistant import WatchList
        from assistant.Memos import Memos
        from assistant.Procedures import Procedures
        from assistant.agents import LongTermMemoryCurator as curator_module
        folders = {Procedures: tmp_path / "procedures", Memos: tmp_path / "memos"}
        watchlist, logs = tmp_path / "watchlist", tmp_path / "logs"
        for folder, text in ((folders[Procedures], "p" * 1024), (folders[Memos], "m" * 2048), (watchlist, "w" * 3072)):
            folder.mkdir()
            (folder / "one.md").write_text(text)

        monitor, tab = _make_monitor()
        with patch.object(Procedures, "DIR", folders[Procedures]), patch.object(Memos, "DIR", folders[Memos]), \
             patch.object(WatchList, "WATCHLIST_DIR", watchlist), patch.object(curator_module, "MEMORY_LOG_DIR", logs), \
             patch.object(MemoryLogMonitor, "LOG_DIR", logs):
            curator_module.LongTermMemoryCurator.save_debug_log("b" * 4096, {"new_memory_size": 4096})
            monitor._scan()

        assert [sizes for _, sizes in tab.memory_log_sizes] == [
            {"bank": 4.0, "procedures": 1.0, "memos": 2.0, "watchlist": 3.0}]

    def test_today_is_the_memory_as_it_stands_whatever_the_nights_snapshot_said(self, tmp_path):
        memory = tmp_path / "memory"
        (memory / "memos").mkdir(parents=True)
        (memory / "bank.md").write_text("b" * 2048)
        (memory / "memos" / "trip.md").write_text("m" * 512)
        (memory / "memos" / "wine.md").write_text("m" * 512)
        logs = tmp_path / "logs"
        logs.mkdir()
        _write(logs, f"{date.today():%Y%m%d}_040000_aaaaaa_memory.md", _snapshot_body(9216, folders={"memos": 9216}))

        monitor, tab = _make_monitor()
        with patch.object(MemoryLogMonitor, "LOG_DIR", logs), \
             patch.object(MemoryLogMonitor, "MEMORY_FILE", memory / "bank.md"):
            monitor._scan()

        assert tab.memory_log_sizes == [(date.today(), {"bank": 2.0, "procedures": 0.0, "memos": 1.0, "watchlist": 0.0})]

    def test_counts_words_of_live_memory_bank(self, tmp_path):
        bank = tmp_path / "memory.md"
        bank.write_text("# Long-term memory bank\n\n---\n- Marcus likes  tea\n- Charlotte plays piano\n")

        monitor, tab = _make_monitor()
        with patch.object(MemoryLogMonitor, "LOG_DIR", tmp_path), \
             patch.object(MemoryLogMonitor, "MEMORY_FILE", bank):
            monitor._scan()

        assert tab.memory_word_count == 9  # markdown markers are not words

    def test_missing_memory_bank_counts_zero_words(self, tmp_path):
        monitor, tab = _make_monitor()
        tab.memory_word_count = 42
        with patch.object(MemoryLogMonitor, "LOG_DIR", tmp_path):
            monitor._scan()

        assert tab.memory_word_count == 0
