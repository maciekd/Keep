import queue

import pytest


@pytest.fixture
def test_vault(tmp_path):
    from common.vault.Vault import Vault
    vault = Vault(tmp_path / "vault.bin", "pw")
    vault.add("api_key", "Telegram api_hash", "0123456789abcdef0123456789abcdef")
    vault.add("phone", "mobile", "+46701740612")
    return vault


@pytest.mark.unit
class TestLogWriter:
    def test_writes_to_all_three_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import common.logging as logging_mod

        # Reset module-level state so we get a fresh writer in tmp_path
        old_queue = logging_mod._write_queue
        old_thread = logging_mod._writer_thread
        logging_mod._write_queue = queue.SimpleQueue()
        logging_mod._writer_thread = None

        try:
            logging_mod._ensure_writer_started()
            logging_mod._write_queue.put(("plain INFO\n", "colored INFO\n", "INFO"))
            logging_mod._write_queue.put(None)  # sentinel to stop
            logging_mod._writer_thread.join(timeout=5)

            assert (tmp_path / "log" / "floxy.log").read_text() == "plain INFO\n"
            assert (tmp_path / "log" / "verbose-floxy.out").read_text() == "colored INFO\n"
            assert (tmp_path / "log" / "floxy.out").read_text() == "colored INFO\n"
        finally:
            logging_mod._write_queue = old_queue
            logging_mod._writer_thread = old_thread

    def test_fine_level_excluded_from_normal_log(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import common.logging as logging_mod

        old_queue = logging_mod._write_queue
        old_thread = logging_mod._writer_thread
        logging_mod._write_queue = queue.SimpleQueue()
        logging_mod._writer_thread = None

        try:
            logging_mod._ensure_writer_started()
            logging_mod._write_queue.put(("plain FINE\n", "colored FINE\n", "FINE"))
            logging_mod._write_queue.put(None)
            logging_mod._writer_thread.join(timeout=5)

            assert (tmp_path / "log" / "floxy.log").read_text() == "plain FINE\n"
            assert (tmp_path / "log" / "verbose-floxy.out").read_text() == "colored FINE\n"
            assert (tmp_path / "log" / "floxy.out").read_text() == ""
        finally:
            logging_mod._write_queue = old_queue
            logging_mod._writer_thread = old_thread

    def test_multiline_message_writes_per_line_in_plain_log(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import common.logging as logging_mod
        from colorama import Fore

        old_queue = logging_mod._write_queue
        old_thread = logging_mod._writer_thread
        old_last = logging_mod.last_log_time
        logging_mod._write_queue = queue.SimpleQueue()
        logging_mod._writer_thread = None

        try:
            logging_mod.disable_file_logging = False
            logging_mod.log("line one\nline two\nline three", Fore.WHITE, "INFO")
            logging_mod._write_queue.put(None)
            logging_mod._writer_thread.join(timeout=5)

            plain_text = (tmp_path / "log" / "floxy.log").read_text()
            plain_lines = [l for l in plain_text.splitlines() if l.strip()]
            assert len(plain_lines) == 3
            assert plain_lines[0].endswith("[INFO] line one")
            assert plain_lines[1].endswith("[INFO] line two")
            assert plain_lines[2].endswith("[INFO] line three")
            # Ensure the full unsplit message never appears in any single line
            for line in plain_lines:
                assert "line one\nline two" not in line
        finally:
            logging_mod.disable_file_logging = True
            logging_mod._write_queue = old_queue
            logging_mod._writer_thread = old_thread
            logging_mod.last_log_time = old_last

    def test_shutdown_drains_queue(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import common.logging as logging_mod

        old_queue = logging_mod._write_queue
        old_thread = logging_mod._writer_thread
        logging_mod._write_queue = queue.SimpleQueue()
        logging_mod._writer_thread = None

        try:
            logging_mod._ensure_writer_started()
            logging_mod._write_queue.put(("line1\n", "cline1\n", "INFO"))
            logging_mod._write_queue.put(("line2\n", "cline2\n", "TRACE"))
            logging_mod.shutdown_logging()

            assert not logging_mod._writer_thread.is_alive()
            assert (tmp_path / "log" / "floxy.log").read_text() == "line1\nline2\n"
        finally:
            logging_mod._write_queue = old_queue
            logging_mod._writer_thread = old_thread


@pytest.fixture
def no_log_masker():
    import common.logging as logging_mod
    old = logging_mod._masker
    logging_mod.set_log_masker(None)
    try:
        yield logging_mod
    finally:
        logging_mod._masker = old


@pytest.mark.unit
class TestLogMasker:
    """A vault value restored into a command line (`config_set` of an api_hash) or a reply
    must never reach floxy.out, the console echo or a dump in clear."""

    def test_logged_lines_and_the_console_echo_are_masked(self, no_log_masker, test_vault, capsys):
        from common.vault.Redactor import Redactor
        logging_mod = no_log_masker
        logging_mod.set_log_masker(Redactor(test_vault).mask)
        old_disable = logging_mod.disable_file_logging
        old_queue = logging_mod._write_queue
        logging_mod.disable_file_logging = False
        logging_mod._write_queue = queue.SimpleQueue()
        old_thread = logging_mod._writer_thread
        logging_mod._writer_thread = object()   # never started: the queue is read here
        try:
            logging_mod.note("Running command: config_set(telegram_feed.api_hash, 0123456789abcdef0123456789abcdef)")
            logging_mod.info('`telegram_login("+46701740612")` and 070-174 06 12')
            entries = []
            while not logging_mod._write_queue.empty():
                entries.append(logging_mod._write_queue.get())
        finally:
            logging_mod.disable_file_logging = old_disable
            logging_mod._write_queue = old_queue
            logging_mod._writer_thread = old_thread
        plain = "".join(e[0] for e in entries)
        colored = "".join(e[1] for e in entries)
        echoed = capsys.readouterr().out
        for text in (plain, colored, echoed):
            assert "0123456789abcdef0123456789abcdef" not in text
            assert "46701740612" not in text and "174 06 12" not in text
            assert "config_set(telegram_feed.api_hash, ••••••••)" in text
            assert '`telegram_login("•••• 0612")` and •••• 0612' in text

    def test_a_log_call_from_inside_the_masker_does_not_recurse_or_deadlock(self, no_log_masker):
        logging_mod = no_log_masker
        seen = []

        def masker(text):
            seen.append(text)
            logging_mod.info("inner " + text)     # the redactor logs when it rebuilds its pattern
            return text.replace("secret", "••••")

        logging_mod.set_log_masker(masker)
        old_disable = logging_mod.disable_file_logging
        logging_mod.disable_file_logging = True
        try:
            assert logging_mod.masked_for_log("a secret") == "a ••••"
        finally:
            logging_mod.disable_file_logging = old_disable
        assert seen == ["a secret"]              # the inner line passed through unmasked

    def test_a_failing_masker_keeps_the_line_and_says_so(self, no_log_masker):
        def masker(text):
            raise RuntimeError("boom")

        no_log_masker.set_log_masker(masker)
        assert no_log_masker.masked_for_log("hello") == "hello [log masking failed: RuntimeError: boom]"


@pytest.mark.unit
class TestOnlyTheServerProcessWritesLogFiles:
    """A probe script run from the repository root once appended its lines to the live
    server's log; file logging is on only for a process started as floxy_ai_server.py."""

    @staticmethod
    def _file_logging_disabled(monkeypatch, argv) -> bool:
        import importlib.util
        import sys
        import common.logging as logging_mod
        monkeypatch.setattr(sys, "argv", argv)
        spec = importlib.util.spec_from_file_location("logging_fresh_import", logging_mod.__file__)
        fresh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fresh)
        return fresh.disable_file_logging

    def test_the_server_launcher_logs_to_files_wherever_it_is_started_from(self, monkeypatch):
        assert not self._file_logging_disabled(monkeypatch, ["floxy_ai_server.py", "--port", "8800"])
        assert not self._file_logging_disabled(monkeypatch, ["/srv/keep.example/floxy_ai_server.py"])

    @pytest.mark.parametrize("argv", [["probe_watch_list.py"], ["-c"], [""], [],
                                      ["/srv/keep.example/tui.py"], ["not_floxy_ai_server.py.bak"]])
    def test_any_other_process_importing_the_tree_does_not(self, monkeypatch, argv):
        assert self._file_logging_disabled(monkeypatch, argv)
