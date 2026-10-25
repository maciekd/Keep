#!/usr/bin/env python3
import codecs
import re
import socket
import sys
import time
import select
import termios
import tty
import os
import signal
import threading
from shutil import get_terminal_size

HOST = sys.argv[1] if len(sys.argv) > 1 else "localhost"   # tui.py <host> for a server on another machine

# Terminal escape sequences for cleanup
CSI = "\x1b["
DISABLE_MOUSE_TRACKING = f"{CSI}?1006l{CSI}?1000l"
POP_KEY_REPORTING = f"{CSI}<u"   # undo the server's kitty keyboard protocol push
DISABLE_BRACKETED_PASTE = f"{CSI}?2004l"
EXIT_ALT_SCREEN = f"{CSI}?1049l"
SHOW_CURSOR = f"{CSI}?25h"
RESET = f"{CSI}0m"
SET_TITLE = f"\x1b]0;Keep - {HOST}\x07"   # the terminal tab's title while the TUI runs
CLEAR_TITLE = "\x1b]0;\x07"                # an empty title hands the tab back to the shell's own naming
PORT = 5151
BYE = "\nDisconnected from the server — the server keeps running, only the TUI closed. Run ./console.sh to reconnect. Run stop.sh to stop server."

# Only sleep if running in Terminus terminal
if os.environ.get("TERM_PROGRAM") == "Terminus":
    print("Sleeping so user can remove on-screen keyboard...")
    time.sleep(3)

running = True
sys.stdout.write(SET_TITLE)
sys.stdout.flush()


def cleanup_terminal():
    """Reset terminal state - disable mouse tracking, exit alt screen, show cursor, drop the title"""
    sys.stdout.write(DISABLE_MOUSE_TRACKING + DISABLE_BRACKETED_PASTE + POP_KEY_REPORTING + EXIT_ALT_SCREEN
                     + SHOW_CURSOR + RESET + CLEAR_TITLE)
    sys.stdout.flush()


def draw_reconnect_overlay(attempt):
    """Draw a centered red box showing reconnect status"""
    size = get_terminal_size()
    w, h = size.columns, size.lines

    msg = f" No connection, reconnect attempt #{attempt}... "
    box_w = len(msg) + 4
    box_h = 5
    x = max(1, (w - box_w) // 2 + 1)
    y = max(1, (h - box_h) // 2 + 1)

    RED_BG = f"{CSI}41m"
    WHITE_BOLD = f"{CSI}1;97m"
    HIDE_CURSOR = f"{CSI}?25l"

    top    = "┌" + "─" * (box_w - 2) + "┐"
    empty  = "│" + " " * (box_w - 2) + "│"
    middle = "│ " + msg.center(box_w - 4) + " │"
    bottom = "└" + "─" * (box_w - 2) + "┘"

    out = HIDE_CURSOR
    for i, line in enumerate([top, empty, middle, empty, bottom]):
        out += f"{CSI}{y + i};{x}H{RED_BG}{WHITE_BOLD}{line}{RESET}"
    out += f"{CSI}{h};1H"
    sys.stdout.write(out)
    sys.stdout.flush()


def signal_handler(sig, frame):
    global running
    running = False
    cleanup_terminal()
    print(BYE)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Track resize events
resize_event = threading.Event()
last_dimensions = None


def winch_handler(sig, frame):
    """Handle terminal window size change"""
    global resize_event
    resize_event.set()


signal.signal(signal.SIGWINCH, winch_handler)


KITTY_KEY = re.compile(rb"\x1b\[99(?::\d*)*;(\d+)(?::\d+)?u")
LOCK_MODIFIERS = 64 | 128   # Caps Lock, Num Lock — the terminal reports them with the key
CTRL = 4


def is_kitty_ctrl_c(data):
    """Ctrl+C as the kitty keyboard protocol reports it, whatever lock keys are on."""
    match = KITTY_KEY.match(data)
    return bool(match) and (int(match.group(1)) - 1) & ~LOCK_MODIFIERS == CTRL


def get_terminal_dimensions():
    size = get_terminal_size()
    return f"{size.lines} {size.columns}"


def handle_connection():
    """Returns True if a connection was established (should reset attempt counter)"""
    global last_dimensions, resize_event
    sock = None
    connected = False
    try:
        # Create socket and connect
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)  # Connection timeout
        sock.connect((HOST, PORT))
        sock.settimeout(0.1)  # Short timeout for recv
        connected = True

        # Send initial terminal dimensions, plus the terminal program so the server can
        # pick a rendering the terminal copes with (Terminal.app gets no half-block pixels,
        # Ghostty, iTerm2, WezTerm and kitty get pictures through the kitty graphics
        # protocol). kitty sets no TERM_PROGRAM, and ssh carries TERM but not TERM_PROGRAM
        # to the far side, so TERM (`xterm-kitty`, `xterm-ghostty`) names the terminal then.
        initial_dims = get_terminal_dimensions()
        terminal = os.environ.get("TERM_PROGRAM", "").split(" ")[0]
        if not terminal:
            term = os.environ.get("TERM", "")
            terminal = next((name for name in ("kitty", "ghostty") if name in term), "")
        sock.send((f"{initial_dims} {terminal}".rstrip() + "\n").encode())
        last_dimensions = initial_dims

        # Incremental decoder buffers trailing incomplete UTF-8 sequences
        # across recv() calls, preventing split multi-byte chars from
        # becoming replacement characters
        utf8_decoder = codecs.getincrementaldecoder('utf-8')('replace')

        # Save terminal settings and set to raw mode
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin)

        try:
            while running:
                # Check if terminal was resized
                if resize_event.is_set():
                    resize_event.clear()
                    new_dims = get_terminal_dimensions()
                    if new_dims != last_dimensions:
                        # Send resize command with new dimensions
                        resize_msg = f"RESIZE {new_dims}\n"
                        try:
                            sock.send(resize_msg.encode())
                            last_dimensions = new_dims
                        except (socket.error, OSError):
                            # Connection lost
                            break

                # Check for data from server or stdin
                readable, _, _ = select.select([sock, sys.stdin], [], [], 0.1)

                # Handle data from server
                if sock in readable:
                    try:
                        data = sock.recv(4096)
                        if not data:
                            # Connection closed
                            break
                        sys.stdout.write(utf8_decoder.decode(data))
                        sys.stdout.flush()
                    except socket.timeout:
                        pass
                    except (socket.error, OSError):
                        # Connection lost
                        break

                # Handle keyboard input
                if sys.stdin in readable:
                    # Use os.read for raw terminal - reads all available bytes
                    data = os.read(sys.stdin.fileno(), 32)
                    if not data:
                        continue

                    # Ctrl+C: ASCII 3, or `CSI 99;<mods>u` once the server has the terminal
                    # reporting keys the kitty way
                    if data[0] == 3 or is_kitty_ctrl_c(data):
                        raise KeyboardInterrupt

                    try:
                        sock.send(data)
                    except (socket.error, OSError):
                        # Connection lost
                        break

        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    except socket.timeout:
        pass  # Connection timeout, will retry
    except socket.error:
        pass  # Connection failed, will retry
    except KeyboardInterrupt:
        # Ctrl+C pressed
        raise
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        if sock:
            sock.close()
    return connected


# Main loop with reconnection
try:
    attempt = 0
    while running:
        try:
            was_connected = handle_connection()
        except KeyboardInterrupt:
            running = False
            cleanup_terminal()
            print(BYE)
            break

        if running:
            if was_connected:
                attempt = 0
            attempt += 1
            draw_reconnect_overlay(attempt)
            time.sleep(1)
        else:
            break
except KeyboardInterrupt:
    cleanup_terminal()
    print(BYE)
    sys.exit(0)
