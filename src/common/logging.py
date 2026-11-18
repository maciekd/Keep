import os
import queue
import sys
import threading
from datetime import datetime

from colorama import Fore, Style

log_lock = threading.Lock()
last_log_time = None
# Only the server process writes log/: anything else importing the tree from the repository
# root (a test, a one-off script, a Claude Code probe) would append to the live server's files.
disable_file_logging = os.path.basename(sys.argv[0] if sys.argv else "") != "floxy_ai_server.py"
separator_color = '\033[38;2;80;80;80m'  # Darkest gray for │ separators

_write_queue = queue.SimpleQueue()
_writer_thread = None

# Installed by the launcher once the vault exists (`Redactor.mask`): every log line, the
# console echo and the on-disk dumps (`log/prompts`, `log/command_replies`) go through it,
# so a vault value restored into a command line or a reply never reaches the logs in clear.
_masker = None
_masking = threading.local()


# Installed by the telemetry while it runs: every `error()` is handed on with the exception it
# was logged for, so a failure that was caught and survived is still reported.
_error_sink = None
_reporting = threading.local()


def set_error_sink(sink):
    """`sink(message, exception_or_None, handled, outside)` called for every `error()`; None
    removes it. *handled* is false for an exception nothing caught, which a hook handed to
    `error()`; *outside* is what the call site said of the cause."""
    global _error_sink
    _error_sink = sink


def _report_error(message, exception, outside):
    """Hands an error to the sink. An `error()` the sink itself logs is not handed on again,
    and a sink that fails costs one stdout line, never the caller."""
    sink = _error_sink
    if sink is None or getattr(_reporting, "active", False):
        return
    _reporting.active = True
    try:
        if exception is not None:
            sink(message, exception, False, outside)
        else:
            sink(message, sys.exc_info()[1], True, outside)
    except Exception as e:
        print(f"Error sink failed: {type(e).__name__}: {e}")
    finally:
        _reporting.active = False


def set_log_masker(masker):
    """`masker(text) -> text` applied to everything logged; None removes it."""
    global _masker
    _masker = masker


def masked_for_log(text: str) -> str:
    """`text` with the vault's secrets in their masked form, or unchanged while no masker
    is installed. Re-entrant-safe: a log call the masker itself makes (the redactor logs
    when it rebuilds its pattern) passes through untouched instead of recursing."""
    masker = _masker
    if masker is None or getattr(_masking, "active", False):
        return text
    _masking.active = True
    try:
        return masker(text)
    except Exception as e:
        return f"{text} [log masking failed: {type(e).__name__}: {e}]"
    finally:
        _masking.active = False


def _log_writer():
    """Background thread that handles all log file I/O."""
    os.makedirs('log', exist_ok=True)
    log_file = open('log/floxy.log', 'a')
    verbose_file = open('log/verbose-floxy.out', 'a')
    normal_file = open('log/floxy.out', 'a')

    while True:
        entry = _write_queue.get()
        if entry is None:
            break
        plain_line, colored_line, level = entry
        log_file.write(plain_line)
        log_file.flush()
        verbose_file.write(colored_line)
        verbose_file.flush()
        if level != "FINE":
            normal_file.write(colored_line)
            normal_file.flush()

    log_file.close()
    verbose_file.close()
    normal_file.close()


def _ensure_writer_started():
    global _writer_thread
    if _writer_thread is not None:
        return
    _writer_thread = threading.Thread(target=_log_writer, name="LogWriter", daemon=True)
    _writer_thread.start()


def shutdown_logging():
    """Drain the write queue and stop the writer thread."""
    if _writer_thread is not None:
        _write_queue.put(None)
        _writer_thread.join(timeout=5)


def log(message, color, level, source=None):
    global last_log_time
    # Before the lock: the masker may log (see masked_for_log) and log_lock is not re-entrant.
    message = masked_for_log(message)
    with log_lock:
        thread_name_width = 16

        now = datetime.now()
        time_str = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        base_time_str_len = len(time_str)  # Save length before adding color codes

        left_hand_color = Style.DIM + Fore.LIGHTBLACK_EX
        if source is not None:
            left_hand_color = Style.DIM + Fore.YELLOW

        if last_log_time is not None:
            seconds = (now - last_log_time).total_seconds()
            if seconds > 1.0:
                color_time_diff = Style.DIM + Fore.WHITE
                if source is not None:
                    color_time_diff = Fore.LIGHTYELLOW_EX
                if seconds > 9.9:
                    time_str += color_time_diff + " +9.999" + left_hand_color
                else:
                    time_str += color_time_diff + f" +{seconds:.3f}" + left_hand_color
            else:
                if seconds > 9.9:
                    time_str += " +9.999"
                else:
                    time_str += f" +{seconds:.3f}"

            # print an empty line for each minute passed, this is a nice way for a human to intuitively understand time passed in the logs
            minutes_passed = int(seconds // 60)
            # Account for the time diff text " +9.999" which is 7 characters
            blank_line_width = base_time_str_len + 9
            for _ in range(minutes_passed):
                print(left_hand_color + " " * blank_line_width + separator_color + " │ " + left_hand_color + " " * thread_name_width + separator_color + " │ ")

        else:
            time_str += " +0.000"

        last_log_time = now
        if source is not None:
            t = source
        else:
            t = (threading.current_thread().name.replace(" (process_request_thread)", "")).replace("MainThread",
                                                                                                   "Main").replace(
                " (run_one_full_testsuite)", "")

        t = t.removeprefix("Thread-")
        t = t.replace("(", "").replace(")", "")
        t = t.replace("_", " ")
        t = t.replace("  ", " ")
        t = t[:thread_name_width]  # Clip to 16 characters
        t = t.ljust(thread_name_width)  # Pad to 16 characters
        t = t.lower()

        # Map level to single character
        level_char = level[0].upper()  # E, I, T, F, N, etc.

        for m in message.split("\n"):
            colored_line = left_hand_color + level_char + separator_color + "│" + left_hand_color + time_str + separator_color + " │ " + left_hand_color + t + separator_color + " │ " + Style.RESET_ALL + color + m
            print(colored_line)
            if not disable_file_logging:
                _ensure_writer_started()
                plain_line = time_str + " [" + level + "] " + m + "\n"
                _write_queue.put((plain_line, colored_line + "\n", level))


def external_info(source, message):
    log(message, Style.DIM + Fore.LIGHTYELLOW_EX, "INFO", source)


def external_trace(source, message):
    message = "\n".join("  " + line for line in message.split("\n"))
    log(message, Style.DIM + Fore.YELLOW, "TRACE", source)


def external_fine(source, message):
    dark_yellow = '\033[38;2;110;102;64m'  # RGB: darker yellow
    message = "\n".join("  " + line for line in message.split("\n"))
    log(message, dark_yellow, "FINE", source)


def info(message):
    log(message, Style.RESET_ALL, "INFO")


def note(message):
    log(message, Style.BRIGHT + Fore.LIGHTBLUE_EX, "NOTE")


def error(message, exception=None, outside=False):
    """*exception* is for the hooks that log an exception nothing caught. An error logged in
    an `except` block needs no argument: the exception being handled is found by itself.
    *outside* says the cause is not the code — a service away, an account signed out — where
    the exception's class cannot say so (`assistant/ErrorKind.py`): the log line is the same,
    the error tracking gets no issue for it."""
    log(message, Style.BRIGHT + Fore.RED, "ERROR")
    _report_error(message, exception, outside)


def trace(message):
    message = "\n".join("  " + line for line in message.split("\n"))
    log(message, Style.DIM + Fore.WHITE, "TRACE")


def fine(message):
    dark_gray = '\033[38;2;80;80;80m'  # RGB: darker gray
    message = "\n".join("  " + line for line in message.split("\n"))
    log(message, dark_gray, "FINE")
