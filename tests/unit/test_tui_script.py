"""tui.py, the terminal client: the keys it answers itself. It is a script with side effects on
import (the terminal title, the signal handlers), so the pieces under test are lifted out of its
source and compiled on their own."""
import ast
import re
from pathlib import Path

import pytest

TUI_PY = Path(__file__).resolve().parents[2] / 'tui.py'
LIFTED = {"KITTY_KEY", "LOCK_MODIFIERS", "CTRL", "is_kitty_ctrl_c"}


def _lifted():
    tree = ast.parse(TUI_PY.read_text())
    nodes = [node for node in tree.body
             if (isinstance(node, ast.FunctionDef) and node.name in LIFTED)
             or (isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in LIFTED for t in node.targets))]
    namespace = {"re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TUI_PY), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def is_kitty_ctrl_c():
    return _lifted()["is_kitty_ctrl_c"]


@pytest.mark.unit
class TestKittyCtrlC:
    """Once the server has the terminal reporting keys the kitty way, Ctrl+C arrives as
    `CSI 99;<mods>u`, and Caps Lock (+64) and Num Lock (+128) ride along in the modifier
    number. With a lock key on, Ctrl+C used to be forwarded to the server and ignored, and
    over ssh there was no way to leave the TUI (#121)."""

    @pytest.mark.parametrize("data", [
        b"\x1b[99;5u",          # Ctrl alone
        b"\x1b[99;69u",         # Ctrl + Caps Lock
        b"\x1b[99;133u",        # Ctrl + Num Lock
        b"\x1b[99;197u",        # Ctrl + both locks
        b"\x1b[99;5:1u",        # with the press event type
        b"\x1b[99:67;5u",       # with the shifted key reported alongside
        b"\x1b[99;69u\x1b[A",   # followed by more input in the same read
    ])
    def test_ctrl_c_is_recognised_whatever_lock_keys_are_on(self, is_kitty_ctrl_c, data):
        assert is_kitty_ctrl_c(data)

    @pytest.mark.parametrize("data", [
        b"\x1b[99;1u",          # a plain c
        b"\x1b[99;65u",         # c with Caps Lock on: a capital C, not a quit
        b"\x1b[99;6u",          # Ctrl+Shift+C
        b"\x1b[99;7u",          # Ctrl+Alt+C
        b"\x1b[99;3u",          # Alt+C
        b"\x1b[100;5u",         # Ctrl+D
        b"c",
        b"\x1b[A",
    ])
    def test_any_other_key_goes_to_the_server(self, is_kitty_ctrl_c, data):
        assert not is_kitty_ctrl_c(data)
