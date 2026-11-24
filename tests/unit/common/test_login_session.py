"""LoginSession: the login a process started in, told by its audit session id, and whether it has ended."""
import os
import plistlib
from unittest.mock import Mock, patch

import pytest

from common import LoginSession, Platform


def logins(*mine, returncode=0):
    """The Mac with these logins of this user beside another user's."""
    sessions = [{"kCGSSessionUserIDKey": os.getuid() + 1, "kCGSSessionAuditIDKey": 9001}]
    sessions += [{"kCGSSessionUserIDKey": os.getuid(), "kCGSSessionAuditIDKey": asid} for asid in mine]
    answer = Mock(returncode=returncode, stdout=plistlib.dumps({"IOConsoleUsers": sessions}))
    return patch.object(LoginSession.subprocess, "run", return_value=answer)


def unreadable():
    return patch.object(LoginSession.subprocess, "run", side_effect=OSError("no ioreg"))


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(LoginSession, "LOGIN_SESSION_FILE", tmp_path / "cache" / "login_session")
    monkeypatch.setattr(LoginSession, "_started_in", None)
    monkeypatch.setattr(Platform, "current", lambda: Platform.MAC)
    return LoginSession


@pytest.mark.unit
class TestCurrent:
    def test_only_this_users_logins_count(self):
        with logins(4100):
            assert LoginSession.current() == ["4100"]

    def test_a_logged_out_user_is_told_from_logins_that_cannot_be_read(self):
        with logins():
            assert LoginSession.current() == []
        with unreadable():
            assert LoginSession.current() is None
        with logins(4100, returncode=1):
            assert LoginSession.current() is None
        with patch.object(LoginSession.subprocess, "run", return_value=Mock(returncode=0, stdout=b"not a plist")):
            assert LoginSession.current() is None


@pytest.mark.unit
class TestEnded:
    def test_a_process_in_the_current_login_has_not_ended(self, session):
        with logins(4100):
            session.record()
            assert session.ended() == ""
        assert session.LOGIN_SESSION_FILE.read_text().strip() == "4100", "autostart.sh reads the login from the file"

    def test_a_new_login_means_the_one_the_process_started_in_has_ended(self, session):
        with logins(4100):
            session.record()
        with logins(4200):
            assert "4100" in session.ended() and "4200" in session.ended()

    def test_a_user_logged_out_has_ended_the_login(self, session):
        with logins(4100):
            session.record()
        with logins():
            assert session.ended()

    def test_a_second_login_of_the_user_does_not_end_the_first(self, session):
        with logins(4100):
            session.record()
        with logins(4200, 4100):
            assert session.ended() == ""

    def test_logins_that_cannot_be_read_end_nothing(self, session):
        with logins(4100):
            session.record()
        with unreadable():
            assert session.ended() == ""

    def test_a_process_started_without_a_login_records_none_and_never_ends(self, session):
        session.LOGIN_SESSION_FILE.parent.mkdir(parents=True)
        session.LOGIN_SESSION_FILE.write_text("3900\n")
        with logins():
            session.record()
        assert not session.LOGIN_SESSION_FILE.exists(), "an earlier process's login must not stand for this one"
        with logins(4200):
            assert session.ended() == ""

    def test_nothing_is_recorded_on_linux(self, session, monkeypatch):
        monkeypatch.setattr(Platform, "current", lambda: Platform.LINUX)
        with logins(4100) as run:
            session.record()
        run.assert_not_called()
        assert not session.LOGIN_SESSION_FILE.exists() and session.ended() == ""
