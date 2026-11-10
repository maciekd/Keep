from unittest.mock import patch

import pytest

from common import Version


@pytest.fixture(autouse=True)
def fresh_cache():
    clear()
    yield
    clear()


def clear():
    for f in (Version.version, Version.branch, Version.commit, Version.label, Version.detail, Version.release_notes):
        f.cache_clear()


def fake_git(answers):
    """Answers keyed by the git subcommand, or by the full argument tuple where one
    subcommand is asked two things (`rev-parse` gives both the hash and the branch)."""
    return patch.object(Version, "_git", side_effect=lambda *args: answers.get(args, answers.get(args[0])))


BRANCH = ("rev-parse", "--abbrev-ref", "HEAD")


@pytest.mark.unit
def test_version_is_the_commit_month_and_the_commit_count():
    with fake_git({"log": "2026.09", "rev-list": "3868"}):
        assert Version.version() == "v2026.09.3868"


@pytest.mark.unit
def test_detail_adds_the_hash_the_branch_and_flags_local_edits():
    with fake_git({"log": "2026.09", "rev-list": "3868", "rev-parse": "1b6c9ce7", BRANCH: "release", "status": " M file.py"}):
        assert Version.detail() == "v2026.09.3868 (1b6c9ce7, release, dirty)"
    clear()
    with fake_git({"log": "2026.09", "rev-list": "3868", "rev-parse": "1b6c9ce7", BRANCH: "master", "status": ""}):
        assert Version.detail() == "v2026.09.3868 (1b6c9ce7, master)"


@pytest.mark.unit
def test_detail_leaves_the_branch_out_of_a_detached_checkout():
    with fake_git({"log": "2026.09", "rev-list": "3868", "rev-parse": "1b6c9ce7", BRANCH: "HEAD", "status": ""}):
        assert Version.detail() == "v2026.09.3868 (1b6c9ce7)"


@pytest.mark.unit
def test_outside_a_git_checkout_the_version_is_unknown():
    with fake_git({}):
        assert Version.version() == Version.UNKNOWN and Version.detail() == Version.UNKNOWN


@pytest.mark.unit
def test_a_missing_git_binary_is_logged_not_raised():
    with patch.object(Version.subprocess, "run", side_effect=FileNotFoundError("git")):
        assert Version.version() == Version.UNKNOWN


@pytest.mark.unit
def test_label_is_bare_on_release_and_badged_elsewhere():
    base = {"log": "2026.09", "rev-list": "3868"}
    with fake_git({**base, BRANCH: "release"}):
        assert Version.label() == "Keep v2026.09.3868"
    clear()
    with fake_git({**base, BRANCH: "master"}):
        assert Version.label() == "Keep v2026.09.3868 dev"
    clear()
    with fake_git({**base, BRANCH: "fix/tab-bar"}):
        assert Version.label() == "Keep v2026.09.3868 fix/tab-bar"
    clear()
    with fake_git({**base, BRANCH: "HEAD"}):
        assert Version.label() == "Keep v2026.09.3868"


TAG_COMMIT = ("tag", "-l", "--format=%(*objectname)", "v2026.09.3868")
TAG_BODY = ("tag", "-l", "--format=%(contents:body)", "v2026.09.3868")
HEAD = ("rev-parse", "HEAD")
NOTE = "<b>Spotify is back.</b>\n\nSearch works on a fresh app.\n"


@pytest.mark.unit
def test_release_notes_are_the_body_of_the_tag_on_the_running_commit():
    with fake_git({"log": "2026.09", "rev-list": "3868", TAG_COMMIT: "abc123", HEAD: "abc123", TAG_BODY: NOTE}):
        assert Version.release_notes() == NOTE.strip()


@pytest.mark.unit
def test_no_release_notes_when_the_version_tag_names_another_commit():
    with fake_git({"log": "2026.09", "rev-list": "3868", TAG_COMMIT: "abc123", HEAD: "def456", TAG_BODY: NOTE}):
        assert Version.release_notes() == ""


@pytest.mark.unit
def test_no_release_notes_on_an_untagged_commit():
    with fake_git({"log": "2026.09", "rev-list": "3868", TAG_COMMIT: "", HEAD: "abc123", TAG_BODY: ""}):
        assert Version.release_notes() == ""


@pytest.mark.unit
def test_a_tag_without_a_note_gives_no_release_notes():
    with fake_git({"log": "2026.09", "rev-list": "3868", TAG_COMMIT: "abc123", HEAD: "abc123", TAG_BODY: "\n"}):
        assert Version.release_notes() == ""


@pytest.mark.unit
def test_the_commit_is_the_one_the_process_first_saw_whatever_the_checkout_does_later():
    """What a server tells its iMessage satellite to follow: a pull it has not restarted on must not move it."""
    heads = iter(["1b6c9ce7" + "0" * 32, "9f9f9f9f" + "1" * 32])
    with patch.object(Version, "_git", side_effect=lambda *args: next(heads)):
        assert Version.commit() == Version.commit() == "1b6c9ce7" + "0" * 32


@pytest.mark.unit
def test_the_commit_is_read_with_the_version_line_of_the_start_not_by_whoever_asks_first_later():
    """A satellite's host built after the checkout moved (an integration enabled later) must
    still name the commit the process started on."""
    git = {"log": "2026.11", "rev-list": "4100", HEAD: "1b6c9ce7" + "0" * 32, BRANCH: "release", "status": ""}
    with fake_git(git):
        Version.detail()
    with fake_git({**git, HEAD: "9f9f9f9f" + "1" * 32}):
        assert Version.commit() == "1b6c9ce7" + "0" * 32


@pytest.mark.unit
def test_an_unknown_commit_is_empty_not_missing():
    with fake_git({}):
        assert Version.commit() == ""
