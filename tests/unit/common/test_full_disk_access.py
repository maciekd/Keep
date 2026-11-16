"""Full Disk Access: the probe on chat.db and the System Settings pane the grant is made in."""
import sqlite3
from types import SimpleNamespace

import pytest

from common import FullDiskAccess

# The Mac's own behaviour, whatever system the suite runs on (common/Platform).
pytestmark = pytest.mark.usefixtures("on_a_mac")


class TestProbe:
    def test_a_readable_messages_database_means_granted(self, tmp_path):
        db = tmp_path / "chat.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE message (ROWID INTEGER PRIMARY KEY)")
        assert FullDiskAccess.granted(db) is True
        assert FullDiskAccess.observable(db) is True

    def test_a_database_that_cannot_be_opened_means_not_granted(self, tmp_path):
        db = tmp_path / "chat.db"
        db.write_text("not a database")
        assert FullDiskAccess.granted(db) is False

    def test_no_messages_database_means_not_granted_and_not_observable(self, tmp_path):
        db = tmp_path / "chat.db"
        assert FullDiskAccess.granted(db) is False
        assert FullDiskAccess.observable(db) is False


class TestPretendMissing:
    def test_a_readable_database_is_reported_missing_while_pretending(self, tmp_path):
        db = tmp_path / "chat.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE message (ROWID INTEGER PRIMARY KEY)")
        FullDiskAccess.pretend_missing()
        try:
            assert FullDiskAccess.pretending_missing() is True
            assert FullDiskAccess.granted(db) is False
        finally:
            FullDiskAccess.pretend_missing(False)
        assert FullDiskAccess.granted(db) is True


class TestOpenSettings:
    def test_opens_the_privacy_pane_through_the_open_command(self):
        calls = []

        def run(argv, **kwargs):
            calls.append(argv)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        assert FullDiskAccess.open_settings(run) is None
        assert calls[0][0] == "open" and "Privacy_AllFiles" in calls[0][1]

    def test_a_failed_open_is_reported_with_its_words(self):
        def run(argv, **kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="no such URL scheme")
        failure = FullDiskAccess.open_settings(run)
        assert failure and "no such URL scheme" in failure

    def test_a_missing_open_command_is_reported(self):
        def run(argv, **kwargs):
            raise OSError("open: not found")
        failure = FullDiskAccess.open_settings(run)
        assert failure and "not found" in failure
