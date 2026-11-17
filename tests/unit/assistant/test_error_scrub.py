"""What an error message loses before it is reported, and what it keeps."""
from pathlib import Path

import pytest

from assistant.ErrorScrub import scrub


@pytest.mark.parametrize("private", [
    "alice@example.com",
    "m.lab+keep@mlab.test",
    "+46701740605",
    "070-174 06 05",
    "19850203-1234",
    "1234567890:AAF-fictionalBotToken_0123456789abcd",
    "0123456789abcdef0123456789abcdef",
])
def test_what_identifies_or_unlocks_is_blanked(private):
    scrubbed = scrub(f"Feed: failed on {private}, giving up")
    assert private not in scrubbed
    assert scrubbed.startswith("Feed: failed on ") and scrubbed.endswith(", giving up")


@pytest.mark.parametrize("readable", [
    "HTTP 503 from the bridge on port 8898",
    "timed out after 30000 ms at line 412",
    "code 123456 was refused",
    "IntegrationMonitor.check_health raised KeyError: 'zones'",
    "src/assistant/integration/FeedRunner.py",
])
def test_what_makes_an_error_readable_stays(readable):
    assert scrub(readable) == readable


def test_the_home_folder_is_not_named():
    scrubbed = scrub(f"cannot read {Path.home()}/Documents/Keep/config.yaml")
    assert str(Path.home()) not in scrubbed and scrubbed.endswith("/Documents/Keep/config.yaml")
