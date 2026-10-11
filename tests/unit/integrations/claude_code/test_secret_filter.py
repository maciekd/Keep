import pytest

from integrations.claude_code.SecretFilter import scrub, looks_secret, REDACTED


@pytest.mark.unit
class TestScrub:
    @pytest.mark.parametrize("text", [
        "export OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz012345",
        "key sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789 here",
        "token ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "github_pat_11ABCDEFG0abcdefghijklmnopqrstuvwxyz",
        "xoxb-123456789012-abcdefghijklmnop",
        "AKIAIOSFODNN7EXAMPLE",
        "AIzaSyA-abcdefghijklmnopqrstuvwxyz0123456",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123456789",
        "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBg\n-----END PRIVATE KEY-----",
        "password=hunter2hunter2",
        'set "api_key": "Qk9iYWxpY2U3ODkw"',
        "token: abcd1234",
        "VGhpc0lzQVZlcnlMb25nT3BhcXVlVG9rZW4xMjM0NTY3ODkw",
    ])
    def test_a_credential_is_blanked(self, text):
        scrubbed = scrub(text)
        assert REDACTED in scrubbed and looks_secret(text)
        assert "hunter2" not in scrubbed and "Qk9iYWxp" not in scrubbed

    @pytest.mark.parametrize("text", [
        "git commit -m 'fix: the hall lamp'",
        "commit 3f26110cfe3140d0a840fa9f0874de23a2b4c6d8 on feat/lamps",
        "session 75661f71-e022-4425-8e00-d54213d8fd27 in /Users/marcus/Keep/src/integrations/claude_code/SessionLog.py",
        "set the token to the one in the vault",
        "api_key: YOUR_OPENAI_API_KEY_HERE",
        "password: <ask the user>",
        "the sha256 is 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        "https://github.example/maciekd/Keep/pull/42",
    ])
    def test_ordinary_text_stays(self, text):
        assert scrub(text) == text and not looks_secret(text)

    def test_the_key_name_stays_so_the_row_says_what_was_set(self):
        assert scrub("password=hunter2hunter2 and token: abcd1234") == f"password={REDACTED} and token: {REDACTED}"

    def test_empty_text(self):
        assert scrub("") == "" and scrub(None) is None
