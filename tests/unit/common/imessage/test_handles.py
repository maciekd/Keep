import pytest

from common.imessage.Handles import is_group_guid, looks_like_handle, normalize_phone, phone_suffix, prefer_handle


@pytest.mark.unit
class TestPhoneSuffix:
    def test_international_format(self):
        assert phone_suffix("+46701740620") == "701740620"

    def test_local_format(self):
        assert phone_suffix("0701740620") == "701740620"

    def test_short_number_returns_none(self):
        assert phone_suffix("12345") is None


@pytest.mark.unit
class TestNormalizePhone:
    def test_strips_formatting(self):
        assert normalize_phone("+1 (234) 567-890") == "+1234567890"


@pytest.mark.unit
class TestPreferHandle:
    def test_e164_phone(self):
        assert prefer_handle("+1 (234) 567-8901") == "+12345678901"

    def test_email_lowercased(self):
        assert prefer_handle("Alice@Example.COM") == "alice@example.com"


@pytest.mark.unit
class TestLooksLikeHandle:
    def test_email_yes(self):
        assert looks_like_handle("a@b.com") is True

    def test_phone_yes(self):
        assert looks_like_handle("+46701234567") is True

    def test_short_digit_string_no(self):
        assert looks_like_handle("12") is False


@pytest.mark.unit
class TestGroupGuid:
    def test_a_group_chat_guid_is_told_from_a_buddy(self):
        assert is_group_guid("any;+;chat100")
        assert not is_group_guid("any;-;+46701740605")
        assert not is_group_guid("+46701740605")
        assert not is_group_guid(None)
