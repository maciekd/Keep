"""The Accessibility seam over stand-in frameworks: what a failed call becomes, what the session
dictionary means. That nothing of pyobjc is loaded before the first call is seen in a process of
its own (tests/unit/test_satellite_role.py)."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from common.imessage.Accessibility import Accessibility, Session

CANNOT_COMPLETE = -25204        # kAXErrorCannotComplete: the app did not answer


def frameworks(**parts):
    mac = SimpleNamespace(ax=Mock(), quartz=Mock(), appkit=Mock(), cf=Mock(), objc=Mock())
    mac.objc.autorelease_pool.return_value.__enter__ = Mock()
    mac.objc.autorelease_pool.return_value.__exit__ = Mock(return_value=False)
    for name, value in parts.items():
        setattr(mac, name, value)
    return mac


@pytest.mark.unit
class TestTheSession:
    @pytest.mark.parametrize("state, session", [
        ({"kCGSSessionOnConsoleKey": True}, Session(on_console=True, locked=False)),
        ({"kCGSSessionOnConsoleKey": True, "CGSSessionScreenIsLocked": True}, Session(on_console=True, locked=True)),
        ({"kCGSSessionOnConsoleKey": False}, Session(on_console=False, locked=False)),
        ({}, Session(on_console=False, locked=False)),
        (None, None),
    ], ids=["unlocked: the lock key is absent", "locked", "switched away", "no console key", "no session"])
    def test_the_window_servers_dictionary_is_read_as_macos_writes_it(self, state, session):
        """A contract with macOS: the two keys of CGSessionCopyCurrentDictionary."""
        mac = frameworks()
        mac.quartz.CGSessionCopyCurrentDictionary.return_value = state
        assert Accessibility(lambda: mac).session() == session


@pytest.mark.unit
class TestACallThatFails:
    def test_an_element_that_does_not_answer_has_no_attribute_no_action_and_takes_nothing(self):
        mac = frameworks()
        mac.ax.AXUIElementCopyAttributeValue.return_value = (CANNOT_COMPLETE, None)
        mac.ax.AXUIElementCopyActionNames.return_value = (CANNOT_COMPLETE, None)
        mac.ax.AXUIElementSetAttributeValue.return_value = CANNOT_COMPLETE
        mac.ax.AXUIElementPerformAction.return_value = CANNOT_COMPLETE
        ax = Accessibility(lambda: mac)
        assert ax.attribute("element", "AXTitle") is None and ax.actions("element") == []
        assert ax.set_attribute("element", "AXValue", "…") is False and ax.perform("element", "AXPress") is False

    def test_a_call_that_went_through_hands_back_what_the_api_did(self):
        mac = frameworks()
        mac.ax.AXUIElementCopyAttributeValue.side_effect = lambda element, name, _: (0, {"AXTitle": "Family",
                                                                                         "AXChildren": ("a", "b")}[name])
        mac.ax.AXUIElementSetAttributeValue.return_value = 0
        ax = Accessibility(lambda: mac)
        assert ax.attribute("window", "AXTitle") == "Family" and ax.attribute("window", "AXChildren") == ["a", "b"]
        assert ax.set_attribute("field", "AXValue", "…") is True
        mac.ax.AXUIElementSetAttributeValue.assert_called_once_with("field", "AXValue", "…")

    def test_an_app_that_is_not_running_has_no_element(self):
        mac = frameworks()
        mac.appkit.NSRunningApplication.runningApplicationsWithBundleIdentifier_.return_value = []
        assert Accessibility(lambda: mac).application("com.apple.MobileSMS") is None
        mac.ax.AXUIElementCreateApplication.assert_not_called()

    def test_frameworks_that_are_missing_raise_at_the_first_call_not_at_construction(self):
        def missing():
            raise ImportError("No module named 'ApplicationServices'")
        ax = Accessibility(missing)
        with pytest.raises(ImportError):
            ax.trusted()
