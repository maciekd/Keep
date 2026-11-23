"""The Codex-style chat pane on the Home tab: a transcript of `•` bullets above a `›`
composer framed by two thin rules, with a dim footer of key hints.

The pane is the screen side of `ConsoleTransport`: the transport (called from ChatSession
worker threads) writes entries into the model here, and the TUI loop draws them. The model
is guarded by a lock; an entry's text lives in a `TypewriterText`, and only what the assistant
emits — its replies and its thinking — reveals with the CRT sweep; everything else (the user's
turns, command echoes, media labels, prompts, history) shows up at once. A photo (or a video's
poster frame) the assistant sends is drawn in the transcript — `IMAGE_ROWS` of the pane's height,
fitted to the picture's proportions — with its caption under it (`TUIScreen.image`: the
terminal's graphics protocol where it has one, half-block pixels where not), and scrolls with
the text a row at a time. `HistoryFeed` fills the same transcript with the
day's dialog history — other channels' turns, system events — inserted by timestamp and
shown at once, each under a dim header saying who and when. Assistant, notice and system
entries hold the chat HTML subset; user text is plain; a diff entry is a `-`/`+` listing.
Vaulted values are masked the way the Vault tab lists them (`•••• 1234`) as an entry is laid
out — every kind, streaming replies included — and the masks are marked in orange, as in the
Memory tab. A value still arriving in a streaming chunk is masked once it is whole.

Keys: printable text edits the composer (`LineEditor`), Enter sends, Shift+Enter (or Alt+Enter,
for a terminal that cannot tell Shift+Enter from Enter) breaks a line, ↑/↓ scroll the transcript
(a line at a time, as in every other tab), PgUp/PgDn browse the composer's history — only the
entries containing what is typed, so "//feed" + PgUp recalls the last //feed command — Esc
interrupts a running reply or clears the composer. A picture pasted with Ctrl+V (the Home tab
reads the clipboard and calls `attach_image`) is staged above the text as a thumbnail
(`THUMBNAIL_ROWS` high, the staged pictures side by side) with its chip under it —
`[image 1: 1200×800, 340 kB]` — until Enter sends it with the text as its caption
(`on_submit(text, images)`) or Esc drops it; a pane too short for thumbnails shows the chips
alone on one row. The sent turn shows the picture under the `›` with the text as its caption,
as the assistant's photos show theirs.
While an approval prompt is pending the pane is modal: y approves, n or Esc denies. While a
secret prompt is pending the composer becomes a masked field of its own (a history-less
`LineEditor`, so the value never reaches tui_state.json): Enter hands the value on, Tab
reveals it, Esc cancels; the value is never added to the transcript.
"""
import math
import re
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable, List, NamedTuple, Optional

from assistant.commands.CommandEcho import describe_duration
from common import Browser
from common.DraftPlaceholder import progress_bar
from common.logging import info, trace
from common.vault.Redactor import Redactor
from common.vault.Vault import Vault
from interfaces.chat import FallbackNotice
from interfaces.chat.ChatTypes import (DECISION_APPROVE, DECISION_DENY, PHASE_COMMANDS, PHASE_PREFILL, PHASE_QUEUED,
                                       PHASE_THINKING, Progress, describe_rate)
from interfaces.console import Theme
from interfaces.console.SystemEvents import CLASS_DESCRIPTION
from interfaces.console.framework.Pictures import Picture
from interfaces.console.framework.TUIScreen import Color, TUIScreen
from interfaces.console.framework.TextWidth import cells, char_width, chunks, clip, column, display_width, normalize
from interfaces.console.tabs.ScrollableTabView import draw_scrollbar
from interfaces.console.tabs.TelegramHtmlParser import StyledSpan, parse as parse_html, wrap_spans
from interfaces.console.widgets.ChatTable import Table, split_blocks, table_lines
from interfaces.console.widgets.CrtCursor import CrtCursor
from interfaces.console.widgets.DiffText import diff_spans
from interfaces.console.widgets.FeedStrip import (HEADER as CHIP_HEADER, SEPARATOR as CHIP_SEPARATOR, FeedChip,
                                                  format_eta, strip_width)
from interfaces.console.widgets.LineEditor import LineEditor
from interfaces.console.widgets.SecretMarks import mark_masks
from interfaces.console.widgets.TypewriterText import TypewriterText

KIND_USER = "user"
KIND_ASSISTANT = "assistant"
KIND_THINKING = "thinking"
KIND_NOTICE = "notice"        # one-shot sends: notifications, command lines, apologies
KIND_MEDIA = "media"
KIND_APPROVAL = "approval"
KIND_SECRET = "secret"        # a masked prompt for one value (vault_request)
# ↑/↓ scroll the transcript a line at a time; PgUp/PgDn stand in for the arrows the
# `LineEditor` reads as "older" / "newer" in its history, filtered by the composer's text.
SCROLL_KEYS = {'up': 1, 'down': -1}
HISTORY_KEYS = {'pageup': 'up', 'pagedown': 'down'}

KIND_SYSTEM = "system"        # history: home-automation events, iMessage traffic, startup…
KIND_DIFF = "diff"            # history: a prompt-section delta, its -/+ lines drawn as a diff

BULLET = "• "
PROMPT = "› "


@dataclass
class Entry:
    kind: str
    body: TypewriterText
    timestamp: float = field(default_factory=time.time)
    final: bool = True
    token: Optional[str] = None       # approval and secret prompts: the hub's token
    header: Optional[str] = None      # dim line above the body (who, when) for history entries
    detail: Optional[str] = None      # command echoes: the dim "(done after 4s, …)" note after the body
    quote: Optional[str] = None       # command echoes: what the command said to somebody else, under the body
    history: bool = False             # a `HistoryFeed` backfill row, not something this console produced live
    debug: bool = False               # a diagnostic row (a cache-break report): drawn only with show_debug on
    picture: Optional[Picture] = None  # media: the photo itself (or a video's poster frame), when it decoded
    _layout: Optional[tuple] = field(default=None, repr=False, compare=False)


@dataclass
class StagedImage:
    """A picture pasted into the composer, waiting to be sent with the next message."""
    data: bytes
    mime_type: str
    picture: Picture

    def chip(self, number: int) -> str:
        return f"[image {number}: {self.picture.width}×{self.picture.height}, {len(self.data) / 1024:.0f} kB]"


class ImageRow(NamedTuple):
    """One row of a picture laid out in a `cols` × `rows` box: the line shows row `row` of it."""
    picture: Picture
    row: int
    cols: int
    rows: int


class Line(NamedTuple):
    """One display line of the transcript. `prefix` is the bullet/prompt on an entry's first
    line and blank on its continuation lines; `entry` is None for the working line; `image`
    makes the line a row of a picture instead of text."""
    prefix: str
    spans: List[StyledSpan]
    entry: Optional[Entry]
    header: bool = False
    image: Optional[ImageRow] = None


class ChatPane:
    MAX_ENTRIES = 500
    MAX_INPUT_ROWS = 3
    MIN_TRANSCRIPT_ROWS = 3
    FOOTER = " ⏎ send  ⇧⏎ newline  ^v paste image  ↑↓ scroll  ⇞⇟ history  esc clear · stop"
    MAX_IMAGES = 10            # pictures one message may carry
    THUMBNAIL_ROWS = 4         # a staged picture's thumbnail above the composer, as tall as the transcript's smallest
    THUMBNAIL_GAP = 2          # columns between two thumbnails
    DEBUG_HINT = "^d debug"
    LINK_HINT = "^o open link"             # bright, while a link is on screen in a terminal without hyperlinks
    CLICK_HINT = "⌘-click opens a link"    # dim, while a link is on screen in a terminal with them
    LINK_OPENED = "Link opened in the browser on the machine Keep runs on."
    LINK_NOT_OPENED = "The link could not be opened on the machine Keep runs on, see the log."
    FALLBACK_NOTICE = FallbackNotice.PATTERN
    # A stripped command leaves a whitespace-only line behind ("text\n\n \n\ntext"): drawn
    # literally that is three empty rows, so any run of blank lines is one paragraph break.
    BLANK_RUN = re.compile(r"\n[ \t]*(?:\n[ \t]*)+")
    APPROVAL_HINT = "approve?  y / n"
    APPROVAL_KEYS = "[y] approve · [n] deny"
    SECRET_FOOTER = " ⏎ save  ⇥ reveal  esc cancel  ↑↓ scroll"
    PROGRESS_LABEL = "Building memory "   # before the footer's progress bar; the first thing dropped when squeezed
    PROGRESS_BAR_WIDTH = 20
    PROGRESS_ETA = " ETA: {}"             # after the bar; dropped after the label, before the bar
    PROGRESS_BAR_AIR = 2                  # cells kept clear on each side of the bar
    SECRET_NOTE_LEAD = "SECURE INPUT"    # right of the masked field, in the label's colour
    SECRET_NOTE_REST = " (stored locally, AI will never see it)"   # dim
    SECRET_NOTE = SECRET_NOTE_LEAD + SECRET_NOTE_REST
    SECRET_NOTE_MIN_FIELD = 16   # cells the field keeps before the note gives way
    MASK = "•"
    SHOW_ELAPSED_AFTER = 2.5   # seconds a request runs before the working line counts them
    PREFILL_BAR_WIDTH = 12     # cells of the working line's prompt-prefill bar
    IMAGE_ROWS = 3             # a picture takes the transcript's rows divided by this…
    IMAGE_ROWS_MIN = 4         # …but no fewer rows than this
    IMAGE_ROWS_MAX = 16        # …and no more than this

    # The chat speaks in the Theme's green and yellow; everything else is one of its text levels.
    COLOR_USER = Theme.COLOR_GREEN
    COLOR_ASSISTANT = Theme.COLOR_TEXT
    COLOR_EMPHASIS = Theme.mix(Theme.COLOR_TEXT, Theme.COLOR_YELLOW, 0.6)   # bold / italic in a reply
    COLOR_INPUT = Theme.COLOR_TEXT
    COLOR_DIM = Theme.COLOR_DIM
    COLOR_LINK_HINT = Theme.COLOR_ORANGE
    COLOR_RULE = Theme.COLOR_FAINT
    COLOR_PROMPT = Theme.mix(Theme.COLOR_PANEL_BG, Theme.COLOR_GREEN, 0.8)
    COLOR_CODE = Theme.COLOR_YELLOW
    COLOR_CHIP = Theme.COLOR_YELLOW    # a staged image's chip above the composer text
    COLOR_LINK = Theme.COLOR_CYAN
    COLOR_QUOTE = Theme.COLOR_LABEL
    COLOR_STRIKE = Theme.mix(Theme.COLOR_DIM, Theme.COLOR_LABEL, 0.5)
    COLOR_THINKING = Theme.COLOR_LABEL
    COLOR_WORKING = Theme.COLOR_YELLOW
    COLOR_APPROVAL = Theme.COLOR_YELLOW
    COLOR_NOTICE_FG = Theme.mix(Theme.COLOR_PANEL_BG, Theme.COLOR_RED, 0.6)
    COLOR_SYSTEM = Theme.COLOR_LABEL
    COLOR_HEADER_DETAIL = Theme.COLOR_FAINT    # the "(channel in location)" part of a header

    def __init__(self, redactor: Optional[Redactor] = None):
        self._redactor = redactor   # over the Vault singleton, resolved on first use
        self._lock = threading.Lock()
        self._entries: List[Entry] = []
        self._working: Optional[Progress] = None
        self._status: Optional[str] = None
        self._pending_approval: Optional[Entry] = None
        self._pending_secret: Optional[Entry] = None
        self._secret_editor: Optional[LineEditor] = None
        self._secret_revealed = False
        self.editor = LineEditor()
        self._images: List[StagedImage] = []       # pasted, not yet sent
        self.scroll_offset = 0
        self.footer_notice: Optional[str] = None   # e.g. "WAITING FOR USER", drawn red at the right
        self.footer_chips: List[FeedChip] = []     # the feed strip (a bar chip per feed), at the right when no notice shows
        self.footer_progress: Optional[float] = None   # 0..1: a bar centred between the hints and the strip
        self.footer_eta: Optional[float] = None        # seconds left, written after the bar
        self.show_debug = False                    # thinking blocks and the fallback notice show only when on
        self._input_cursor = CrtCursor()
        self._last_transcript_rows = 10
        self._min_transcript_rows = 10             # the rows left once the header is fully shown
        self._total_lines = 0
        self._visible_links: List[str] = []       # the links on screen after the last draw, oldest first
        # A header (the Home tab's logo and title) takes `header_rows` rows at the top of the
        # transcript and scrolls away with it: once the lines fill the rows under it, every
        # further line lifts the header a row, `header_scroll` rows in all, until it is gone
        # and the transcript scrolls on its own; scrolling back through the history brings
        # it down again. The owner draws the header itself, lifted by `header_scroll`.
        self.header_rows = 0
        self.header_scroll = 0
        self._last_window = None                   # (first visible line, top row) of the last frame
        self.on_submit: Callable[[str, List[StagedImage]], None] = lambda text, images: trace(
            f"ChatPane: nothing bound to receive {text!r} and {len(images)} image(s)")
        self.on_interrupt: Callable[[], None] = lambda: trace("ChatPane: nothing bound to interrupt")
        self.on_approval: Callable[[str, str], None] = lambda token, decision: trace(
            f"ChatPane: nothing bound to resolve approval {token[:8]} ({decision})")
        # (token, value) — value None when the user cancelled. The value is never logged.
        self.on_secret: Callable[[str, Optional[str]], None] = lambda token, value: trace(
            f"ChatPane: nothing bound to take the secret for {token[:8]}")

    def bind(self, *, on_submit=None, on_interrupt=None, on_approval=None, on_secret=None):
        if on_submit is not None:
            self.on_submit = on_submit
        if on_interrupt is not None:
            self.on_interrupt = on_interrupt
        if on_approval is not None:
            self.on_approval = on_approval
        if on_secret is not None:
            self.on_secret = on_secret

    @property
    def redactor(self) -> Redactor:
        if self._redactor is None:
            self._redactor = Redactor(Vault.get_instance())
            trace(f"ChatPane: secret masking resolved, {len(self._redactor.vault)} vault secret(s)")
        return self._redactor

    # --- model (any thread) ---------------------------------------------------------------

    @property
    def entries(self) -> List[Entry]:
        with self._lock:
            return list(self._entries)

    @property
    def working(self) -> Optional[Progress]:
        return self._working

    @property
    def status(self) -> Optional[str]:
        return self._status

    @property
    def pending_approval(self) -> Optional[Entry]:
        return self._pending_approval

    @property
    def pending_secret(self) -> Optional[Entry]:
        return self._pending_secret

    @property
    def secret_revealed(self) -> bool:
        return self._secret_revealed

    @property
    def images(self) -> List[StagedImage]:
        return list(self._images)

    def attach_image(self, data: bytes, mime_type: str) -> bool:
        """Stage a pasted picture for the next message. Refused — False, logged — while a
        prompt owns the composer, when the picture does not decode, or at MAX_IMAGES."""
        if self._pending_secret is not None or self._pending_approval is not None:
            info("ChatPane: a prompt owns the composer, the pasted image is ignored")
            return False
        if len(self._images) >= self.MAX_IMAGES:
            info(f"ChatPane: {self.MAX_IMAGES} images are staged already, the pasted one is ignored")
            return False
        picture = Picture.decode(data)
        if picture is None:
            info("ChatPane: the pasted image does not decode, ignored")
            return False
        self._images.append(StagedImage(data, mime_type, picture))
        info(f"ChatPane: image staged ({picture.width}×{picture.height}, {len(data) / 1024:.0f} kB), "
             f"{len(self._images)} in the composer")
        return True

    def paste_text(self, text: str) -> bool:
        """Clipboard text into whichever editor has the composer — the masked field while a
        secret prompt owns it; none while an approval does."""
        if not text:
            trace("ChatPane: nothing to paste")
            return False
        if self._pending_approval is not None:
            trace("ChatPane: an approval prompt owns the composer, the pasted text is ignored")
            return False
        editor = self._secret_editor if self._pending_secret is not None else self.editor
        editor.insert(text)
        return True

    def _add(self, entry: Entry) -> Entry:
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self.MAX_ENTRIES:
                del self._entries[:len(self._entries) - self.MAX_ENTRIES]
        return entry

    def add_user(self, text: str, pictures: List[Picture] = ()) -> Entry:
        """The user's turn: its pictures each as an entry of their own, the text as the
        caption of the last one — or on its own when there is none."""
        self.scroll_offset = 0
        pictures = list(pictures)
        for picture in pictures[:-1]:
            self._add(Entry(KIND_USER, TypewriterText("", reveal=False), picture=picture))
        return self._add(Entry(KIND_USER, TypewriterText(text, reveal=False), picture=pictures[-1] if pictures else None))

    def insert(self, kind: str, text: str, timestamp: float, header: Optional[str] = None) -> Entry:
        """A history entry: shown at once (no reveal) and placed by timestamp, after every
        entry that is not newer — so a backdated event lands where it happened."""
        entry = Entry(kind, TypewriterText(text, reveal=False), timestamp=timestamp, header=header, history=True)
        with self._lock:
            index = len(self._entries)
            while index > 0 and self._entries[index - 1].timestamp > timestamp:
                index -= 1
            self._entries.insert(index, entry)
            if len(self._entries) > self.MAX_ENTRIES:
                del self._entries[:len(self._entries) - self.MAX_ENTRIES]
        return entry

    def add_notice(self, html: str) -> Entry:
        return self._add(Entry(KIND_NOTICE, TypewriterText(html, reveal=False)))

    def add_debug_diff(self, text: str, header: str) -> Entry:
        """A diagnostic row (a cache-break report) under a dim header: its -/+ lines are
        drawn as a diff, and the row shows only while show_debug is on."""
        return self._add(Entry(KIND_DIFF, TypewriterText(text, reveal=False), header=header, debug=True))

    def add_media(self, label: str, caption: str, size: int, data: Optional[bytes] = None) -> Entry:
        """A photo or video the assistant sent: the picture (`data`, the photo or a video's
        poster frame) over its caption, or the `[label, size]` line alone when there is no
        picture or it does not decode."""
        picture = Picture.decode(data) if data else None
        if picture is not None:
            text = caption or f"{label}, {size / 1024:.0f} kB"
        else:
            text = f"[{label}, {size / 1024:.0f} kB]"
            if caption:
                text += f" {caption}"
        return self._add(Entry(KIND_MEDIA, TypewriterText(text, reveal=False), picture=picture))

    def add_assistant(self, html: str, final: bool = False) -> Entry:
        return self._add(Entry(KIND_ASSISTANT, TypewriterText(html), final=final))

    def update_entry(self, entry: Entry, html: str, final: bool = False):
        entry.body.set_text(html)
        if final:
            entry.final = True

    def remove_entry(self, entry: Entry):
        with self._lock:
            if entry in self._entries:
                self._entries.remove(entry)
            else:
                trace("ChatPane: entry to remove is no longer in the transcript")

    def set_thinking(self, entry: Optional[Entry], text: str, done: bool = False) -> Entry:
        """Update the live thinking block (creating it on the first chunk)."""
        if entry is None:
            entry = self._add(Entry(KIND_THINKING, TypewriterText(text), final=done))
        else:
            entry.body.set_text(text)
            entry.final = entry.final or done
        return entry

    def set_detail(self, entry: Entry, detail: Optional[str]):
        """Annotate a command echo with how it ended; the body itself stays as it was."""
        entry.detail = detail if detail and detail.strip() else None

    def set_quote(self, entry: Entry, quote: Optional[str]):
        """Put what a command said to somebody else (a `post_message`) under its echo."""
        entry.quote = quote if quote and quote.strip() else None

    def set_working(self, progress: Optional[Progress]):
        self._working = progress

    def set_status(self, html: Optional[str]):
        self._status = html if html and html.strip() else None

    def add_approval(self, token: str, title: str, details: str) -> Entry:
        text = title
        if details:
            text += f"\n{details}"
        entry = self._add(Entry(KIND_APPROVAL, TypewriterText(text, reveal=False), final=False, token=token))
        self._pending_approval = entry
        self.scroll_offset = 0
        return entry

    def close_approval(self, entry: Entry, outcome_text: str):
        first_line = entry.body.text.split("\n", 1)[0]
        entry.body.set_text(f"{first_line} — {_OUTCOME_MARKER.sub('', outcome_text)}")
        entry.final = True
        if self._pending_approval is entry:
            # Two prompts can be open at once (a chat request plus an autonomous action);
            # y/n must then move on to the oldest one still waiting, not leave it orphaned.
            self._pending_approval = self._open_approval()

    def _open_approval(self) -> Optional[Entry]:
        with self._lock:
            return next((e for e in self._entries if e.kind == KIND_APPROVAL and not e.final), None)

    def add_secret_prompt(self, token: str, title: str, details: str) -> Entry:
        """A new prompt, or — same token still open — the existing one re-asked with new
        details (the field stays focused, whatever was typed is already gone)."""
        text = title
        if details:
            text += f"\n{_OUTCOME_MARKER.sub('', details)}"
        entry = self._open_secret(token)
        if entry is not None:
            entry.body.set_text(text)
            trace(f"ChatPane: secret prompt [{token[:8]}] re-asked in place")
        else:
            entry = self._add(Entry(KIND_SECRET, TypewriterText(text, reveal=False), final=False, token=token))
            if self._pending_secret is None:
                self._focus_secret(entry)
        self.scroll_offset = 0
        return entry

    def close_secret(self, entry: Entry, outcome_text: str):
        first_line = entry.body.text.split("\n", 1)[0]
        entry.body.set_text(f"{first_line} — {_OUTCOME_MARKER.sub('', outcome_text)}")
        entry.final = True
        if self._pending_secret is entry:
            self._focus_secret(self._open_secret())

    def _open_secret(self, token: Optional[str] = None) -> Optional[Entry]:
        with self._lock:
            return next((e for e in self._entries if e.kind == KIND_SECRET and not e.final
                         and (token is None or e.token == token)), None)

    def _focus_secret(self, entry: Optional[Entry]):
        """Make `entry` the prompt the composer types into — with a fresh, history-less
        editor — or leave secret mode when there is none."""
        self._pending_secret = entry
        self._secret_editor = LineEditor(persist_history=False) if entry is not None else None
        self._secret_revealed = False
        trace("ChatPane: secret prompt focused" if entry is not None else "ChatPane: secret prompt closed")

    def clear(self):
        with self._lock:
            self._entries.clear()
        self._working = None
        self._status = None
        self._pending_approval = None
        self._focus_secret(None)
        self.scroll_offset = 0

    # --- input (TUI thread) -----------------------------------------------------------------

    def handle_input(self, key: str) -> bool:
        if key == '\x0f':   # Ctrl+O
            self.open_link()
            return True
        if self._pending_secret is not None:
            return self._handle_secret_key(key)
        pending = self._pending_approval
        if pending is not None:
            if key == 'y':
                self._answer_approval(pending, DECISION_APPROVE)
            elif key in ('n', 'escape'):
                self._answer_approval(pending, DECISION_DENY)
            elif key in SCROLL_KEYS:
                return self._scroll(key)
            return True

        if key in SCROLL_KEYS:
            return self._scroll(key)
        if key in HISTORY_KEYS:
            return self.editor.handle_key(HISTORY_KEYS[key])
        if key == 'shift_enter':
            self.editor.insert("\n")
            return True
        if key in ('\r', '\n', 'enter'):
            text = self.editor.take()
            images, self._images = self._images, []
            if text or images:
                self.on_submit(text, images)
            else:
                trace("ChatPane: Enter on an empty composer, nothing to send")
            return True
        if key == 'escape':
            if self._working is not None or self._streaming():
                info("ChatPane: Esc pressed while a reply runs, interrupting it")
                self.on_interrupt()
            elif not self.editor.empty or self._images:
                self.editor.clear()
                if self._images:
                    info(f"ChatPane: Esc drops {len(self._images)} staged image(s)")
                    self._images = []
            return True
        return self.editor.handle_key(key)

    def open_link(self) -> bool:
        """The newest link on screen, opened in the browser of the server's machine: the way
        to follow a link in a terminal that wraps it unclickably. False with none on screen."""
        if not self._visible_links:
            info("ChatPane: Ctrl+O with no link on screen, nothing to open")
            return False
        url = self._visible_links[-1]
        info(f"ChatPane: Ctrl+O opens the newest link on screen ({len(self._visible_links)} visible)")
        opened = Browser.open_url(url)
        # Said in the transcript: over ssh the browser is on a screen the user is not looking at.
        self.add_notice(self.LINK_OPENED if opened else self.LINK_NOT_OPENED)
        return opened

    def _answer_approval(self, entry: Entry, decision: str):
        info(f"ChatPane: approval [{(entry.token or '')[:8]}] answered {decision} from the console")
        self.on_approval(entry.token, decision)

    def _handle_secret_key(self, key: str) -> bool:
        entry, editor = self._pending_secret, self._secret_editor
        token = entry.token or ""
        if key in SCROLL_KEYS:
            return self._scroll(key)
        if key in ('\r', '\n', 'enter'):
            value = editor.text
            if not value.strip():
                trace("ChatPane: Enter on an empty secret field, nothing to hand on")
                return True
            editor.clear()
            self._secret_revealed = False
            info(f"ChatPane: secret [{token[:8]}] entered from the console")
            self.on_secret(entry.token, value)
            return True
        if key == 'escape':
            editor.clear()
            info(f"ChatPane: secret [{token[:8]}] cancelled from the console")
            self.on_secret(entry.token, None)
            return True
        if key in ('tab', '\t'):
            self._secret_revealed = not self._secret_revealed
            trace(f"ChatPane: secret field {'revealed' if self._secret_revealed else 'masked'}")
            return True
        return editor.handle_key(key)

    def _streaming(self) -> bool:
        with self._lock:
            return any(not e.final for e in self._entries if e.kind == KIND_ASSISTANT)

    def _scroll(self, key: str) -> bool:
        max_scroll = max(0, self._total_lines - self._min_transcript_rows)
        self.scroll_offset = min(max_scroll, max(0, self.scroll_offset + SCROLL_KEYS[key]))
        return True

    # --- drawing (TUI thread) ---------------------------------------------------------------

    def draw(self, screen: TUIScreen, frame: int, x: int, y: int, width: int, height: int):
        """Draw the pane into the rectangle whose top-left cell is (x, y)."""
        screen.reset_color()
        footer_y = y + height - 1
        bottom_rule_y = footer_y - 1
        text_w = max(1, width - len(PROMPT) - 1)
        input_rows_all = [""] if self._pending_secret is not None else self._input_rows(text_w)
        input_rows = max(1, min(self.MAX_INPUT_ROWS, len(input_rows_all)))
        staged_rows = self._staged_rows(bottom_rule_y - input_rows - 1 - y)
        top_rule_y = bottom_rule_y - input_rows - staged_rows - 1
        gap_y = top_rule_y - 1              # one blank row keeps the composer off the transcript
        status_y = gap_y - 1 if self._status else None
        transcript_bottom = (status_y if status_y is not None else gap_y) - 1
        transcript_rows = transcript_bottom - y + 1
        if transcript_rows < self.MIN_TRANSCRIPT_ROWS:
            trace(f"ChatPane: only {transcript_rows} transcript row(s) fit in a {height}-row pane")
        self._last_transcript_rows = max(0, transcript_rows)

        self._draw_transcript(screen, frame, x, y, width, transcript_rows)
        if status_y is not None:
            self._draw_status(screen, x, status_y, width)
        screen.write_at(x, gap_y, " " * width)
        self._draw_rule(screen, x, top_rule_y, width)
        if staged_rows:
            self._draw_staged(screen, x, top_rule_y + 1, width, staged_rows)
        self._draw_composer(screen, frame, x, top_rule_y + 1 + staged_rows, width, input_rows, input_rows_all, text_w)
        self._draw_rule(screen, x, bottom_rule_y, width)
        self._draw_footer(screen, x, footer_y, width)

    # transcript ------------------------------------------------------------------------------

    def _draw_transcript(self, screen: TUIScreen, frame: int, x: int, y: int, width: int, rows: int):
        content_w = max(1, width - 3)   # bullet indent and the scrollbar column
        entries = [e for e in self.entries if self.show_debug or not (e.kind == KIND_THINKING or e.debug)]
        image_rows = max(self.IMAGE_ROWS_MIN, min(self.IMAGE_ROWS_MAX, rows // self.IMAGE_ROWS))
        lines = self._layout(entries, content_w, image_rows)
        total = len(lines)
        self._total_lines = total
        if rows <= 0:
            return
        # The header is `header` rows of content above the first line: the window over
        # header + lines ends `scroll_offset` lines from the bottom, and the header rows it
        # starts past are the lifted ones.
        header = max(0, min(self.header_rows, rows - self.MIN_TRANSCRIPT_ROWS))
        lifted = min(header, max(0, header + total - self.scroll_offset - rows))
        self.header_scroll = lifted
        self._min_transcript_rows = rows - header
        y += header - lifted
        rows -= header - lifted
        self._last_transcript_rows = rows
        max_scroll = max(0, total - rows)
        self.scroll_offset = min(self.scroll_offset, max_scroll)
        start = max(0, total - rows - self.scroll_offset)
        visible = lines[start:start + rows]
        self._visible_links = list(dict.fromkeys(s.href for line in visible for s in line.spans if s.href))
        window = (start, y)
        if self._last_window is not None and window != self._last_window:
            # The CRT trails remember screen cells; once the lines under them moved the fade
            # would sit on unrelated text, so every sweep starts over.
            for entry in entries:
                if entry.body.animated:
                    entry.body.reset_cursor()
        self._last_window = window

        for row in range(rows):
            screen.reset_color()
            screen.write_at(x, y + row, " " * width)
        for i, line in enumerate(visible):
            self._draw_line(screen, frame, x, y + i, line, width)
        self._draw_images(screen, x + 2, y, visible)
        if self.redactor.active:
            for i in range(len(visible)):
                mark_masks(screen, self.redactor, x + 2, y + i, width - 3)

        draw_scrollbar(screen, x + width - 1, y, total, rows, self.scroll_offset)
        self._draw_stream_cursors(screen, frame, x, y, visible, entries)

    @staticmethod
    def _draw_images(screen: TUIScreen, x: int, y: int, visible: List[Line]):
        """The pictures among the visible lines: each run of a picture's rows is one band of
        it, drawn at the run's first row."""
        band_start: Optional[ImageRow] = None   # the first row of the band being collected
        band_y = 0                              # the screen row that first row is on
        band_rows = 0
        for i, line in enumerate(visible):
            image = line.image
            continues = (band_start is not None and image is not None and image.picture is band_start.picture
                         and image.row == band_start.row + band_rows)
            if continues:
                band_rows += 1
                continue
            if band_start is not None:
                screen.image(x, band_y, band_start.picture, band_start.cols, band_start.rows,
                             first_row=band_start.row, visible_rows=band_rows)
            band_start, band_y, band_rows = image, y + i, 1
        if band_start is not None:
            screen.image(x, band_y, band_start.picture, band_start.cols, band_start.rows,
                         first_row=band_start.row, visible_rows=band_rows)

    def _layout(self, entries: List[Entry], width: int, image_rows: int) -> List[Line]:
        """Flatten the transcript to display lines; a picture gets `image_rows` rows."""
        lines: List[Line] = []
        for entry in entries:
            entry_lines = self._entry_lines(entry, width, image_rows)
            if not entry_lines:
                continue
            lines.extend(entry_lines)
            # A media label groups with the turn it belongs to; everything else, the thinking
            # block and a picture with its caption included, gets a blank line after it.
            if entry.kind != KIND_MEDIA or entry.picture is not None:
                lines.append(Line("", [], None))
        while lines and not lines[-1].spans:
            lines.pop()
        if self._working is not None:
            if lines:
                lines.append(Line("", [], None))
            lines.append(Line(BULLET, self._working_spans(), None))
        return lines

    def _entry_lines(self, entry: Entry, width: int, image_rows: int) -> List[Line]:
        """An entry's display lines, cached on the entry until its text, width, state or the
        vault changes — a day of history must not be re-masked, re-parsed and re-wrapped
        every frame. The key is the unmasked text: the vault's pattern over every entry costs
        more than a frame's slot, and an animation loop that never sleeps starves the server."""
        visible = entry.body.visible_text()
        if not self.show_debug and entry.kind == KIND_ASSISTANT:
            visible = self._without_fallback_notice(entry.body.text, visible)
        key = (visible, width, image_rows, entry.final, entry.detail, entry.quote, self.redactor.vault.version)
        cached = entry._layout
        if cached is not None and cached[0] == key:
            return cached[1]
        lines = self._wrap_entry(entry, normalize(self.redactor.mask(visible)), width, image_rows)
        entry._layout = (key, lines)
        return lines

    @staticmethod
    def _picture_lines(entry: Entry, text: str, width: int, image_rows: int) -> List[Line]:
        """An entry with a picture: the picture's rows, the entry's mark on the first, then
        the caption under it — dim and italic for a media entry, the user's bold for a
        pasted picture; a user's picture without a caption is the picture alone."""
        user = entry.kind == KIND_USER
        cols, rows = entry.picture.fit(width, image_rows)
        mark = PROMPT if user else BULLET
        lines = [Line(mark if row == 0 else "", [], entry, image=ImageRow(entry.picture, row, cols, rows))
                 for row in range(rows)]
        if user and not text:
            return lines
        for line_spans in wrap_spans([StyledSpan(text=text, bold=user, italic=not user)], width):
            lines.append(Line("", line_spans, entry))
        return lines

    @classmethod
    def _without_fallback_notice(cls, full: str, visible: str) -> str:
        """`visible` (a prefix of `full`, the reveal still running) minus the fallback
        notice. The notice is located in the full text so not even its first characters
        show while the typewriter reaches it."""
        match = cls.FALLBACK_NOTICE.search(full)
        if match is None:
            return visible
        return visible[:match.start()] + visible[match.end():]

    def _wrap_entry(self, entry: Entry, text: str, width: int, image_rows: int) -> List[Line]:
        if entry.picture is not None:
            return self._picture_lines(entry, text.strip(), width, image_rows)
        if not text.strip() and entry.kind not in (KIND_APPROVAL, KIND_SECRET):
            return []
        # The model opens its <thinking> block with a newline and the fallback notice starts
        # with two; drawn literally they leave the bullet alone on an empty row. A diff's
        # lines are literal, every other kind is prose.
        text = text.strip("\n").rstrip()
        if entry.kind != KIND_DIFF:
            text = self.BLANK_RUN.sub("\n\n", text)
        prefix, blocks = self._entry_blocks(entry, text)
        if entry.quote:
            quoted = parse_html(normalize(self.redactor.mask(entry.quote)).strip())
            blocks.append([replace(span, quote=True) for span in quoted])
        lines: List[Line] = []
        if entry.header:
            # A user turn's prompt marks the words, so the header is plain above it; the
            # other kinds carry their bullet on the header line.
            marks_body = prefix == PROMPT
            lines.append(Line("" if marks_body else prefix, _header_spans(entry.header), entry, header=True))
            if not marks_body:
                prefix = ""
        for block in blocks:
            wrapped = (table_lines(block, width, self.COLOR_DIM) if isinstance(block, Table)
                       else wrap_spans(block, width))
            for line_spans in wrapped:
                lines.append(Line(prefix, line_spans, entry))
                prefix = ""
        return lines

    @staticmethod
    def _entry_blocks(entry: Entry, text: str) -> tuple:
        """The entry's prefix and its body as blocks: span lists to word-wrap, and for the
        kinds that carry the chat dialect, the pipe tables among the prose as `Table`s."""
        if entry.kind == KIND_USER:
            return PROMPT, [[StyledSpan(text=text, bold=True)]]
        if entry.kind in (KIND_ASSISTANT, KIND_SYSTEM):
            blocks = [b if isinstance(b, Table) else parse_html(b) for b in split_blocks(text)]
            if entry.detail and blocks and not isinstance(blocks[-1], Table):
                # A system row's verdict (the feed's "rejected: …" on a voice request) trails the body, dim
                blocks[-1].append(StyledSpan(text=f" ({entry.detail})", fg=ChatPane.COLOR_DIM))
            return BULLET, blocks
        if entry.kind == KIND_NOTICE:
            # The hub's command notifications carry a bullet of their own; ours replaces it.
            spans = _lstrip_bullet(parse_html(text))
            if entry.detail:
                spans.append(StyledSpan(text=f" ({entry.detail})", fg=ChatPane.COLOR_DIM))
            return BULLET, [spans]
        if entry.kind in (KIND_THINKING, KIND_MEDIA):
            return BULLET, [[StyledSpan(text=text, italic=True)]]
        if entry.kind == KIND_DIFF:
            return BULLET, [diff_spans(text)]
        spans = [StyledSpan(text=text, bold=not entry.final)]
        if not entry.final and entry.kind == KIND_APPROVAL:
            spans.append(StyledSpan(text="\n" + ChatPane.APPROVAL_KEYS))
        return BULLET, [spans]

    def _working_spans(self) -> List[StyledSpan]:
        """What is being waited for — the LLM ("Thinking…", or "Reading…" with a bar while
        a local model is still on the prompt) or a command ("Running command…") — in the
        working colour, the detail in parentheses dim."""
        p = self._working
        parts = []
        if p.phase == PHASE_THINKING and p.output_tokens:
            parts.append(f"↓ {p.output_tokens / 1000:.1f}k tokens" if p.output_tokens >= 1000
                         else f"↓ {p.output_tokens} tokens")
        if p.phase == PHASE_PREFILL:
            parts.append(f"{p.prefill:.0f}%")
        if p.phase in (PHASE_PREFILL, PHASE_THINKING) and p.rate > 0:
            parts.append(describe_rate(p.rate))
        if p.elapsed >= self.SHOW_ELAPSED_AFTER:
            parts.append(describe_duration(p.elapsed))
        if p.phase == PHASE_COMMANDS:
            label = "Running commands…" if p.commands > 1 else "Running command…"
        elif p.phase == PHASE_PREFILL:
            label = "Reading…"
        elif p.phase == PHASE_QUEUED:
            label = "Waiting in queue…"
        else:
            label = "Thinking…"
        spans = [StyledSpan(text=label, bold=True)]
        if p.phase == PHASE_PREFILL:
            filled, empty = progress_bar(p.prefill / 100.0, self.PREFILL_BAR_WIDTH)
            spans.append(StyledSpan(text=" " + filled, fg=self.COLOR_WORKING))
            spans.append(StyledSpan(text=empty, fg=self.COLOR_DIM))
        if parts:
            spans.append(StyledSpan(text=f" ({' · '.join(parts)})", fg=self.COLOR_DIM))
        return spans

    def _span_fg(self, entry: Optional[Entry], span: StyledSpan) -> Color:
        if span.fg is not None:
            return span.fg
        if entry is None:
            return self.COLOR_WORKING
        if entry.kind == KIND_USER:
            return self.COLOR_USER
        if entry.kind == KIND_THINKING:
            return self.COLOR_THINKING
        if entry.kind == KIND_MEDIA:
            return self.COLOR_DIM
        if entry.kind in (KIND_APPROVAL, KIND_SECRET):
            return self.COLOR_APPROVAL
        if entry.kind in (KIND_SYSTEM, KIND_DIFF):
            if span.bold:   # a message's parties
                return self.COLOR_ASSISTANT
            if span.cls == CLASS_DESCRIPTION:   # a media message's full description
                return self.COLOR_DIM
            return self.COLOR_SYSTEM
        if span.code or span.pre:
            return self.COLOR_CODE
        if span.href is not None:
            return self.COLOR_LINK
        if span.quote:
            return self.COLOR_QUOTE
        if span.strike:
            return self.COLOR_STRIKE
        if span.bold or span.italic:
            return self.COLOR_EMPHASIS
        return self.COLOR_ASSISTANT

    def _draw_line(self, screen: TUIScreen, frame: int, x: int, y: int, line: Line, width: int):
        screen.reset_color()
        if line.prefix:
            screen.set_color(self.COLOR_USER if line.prefix == PROMPT else self.COLOR_DIM)
            screen.bold()
            screen.write_at(x, y, line.prefix)
        cx = x + 2
        for span in line.spans:
            if not span.text:
                continue
            screen.reset_color()
            if line.header:
                fg = self.COLOR_HEADER_DETAIL if span.text.startswith(" (") else self.COLOR_DIM
            else:
                fg = self._span_fg(line.entry, span)
            screen.set_color(fg)
            if span.spoiler:
                screen.set_bg_color(fg)
            elif span.bg is not None:
                screen.set_bg_color(span.bg)
            if span.bold:
                screen.bold()
            if span.italic:
                screen.italic()
            if span.underline or span.href is not None:
                screen.underline()
            if span.href is not None:
                screen.link(span.href)
            if line.entry is None and span.fg is None:
                self._draw_shimmer(screen, frame, cx, y, span.text, fg)
            else:
                screen.write_at(cx, y, span.text)
            cx += display_width(span.text)
        screen.reset_color()
        if line.entry is not None and line.entry.kind == KIND_DIFF and line.spans and line.spans[0].bg is not None:
            screen.tint_row(x + 2, y, width - 3, line.spans[0].bg)   # up to the scrollbar column

    @staticmethod
    def _draw_shimmer(screen: TUIScreen, frame: int, x: int, y: int, text: str, base: Color):
        """The working label breathes: a glow sweeps left to right along the characters,
        with two slower ripples layered on top so it never repeats visibly."""
        col = 0
        for i, ch in enumerate(text):
            screen.set_color(shimmer(base, frame, i))
            screen.bold()
            screen.write_at(x + col, y, ch)
            col += char_width(ch)

    def _draw_stream_cursors(self, screen: TUIScreen, frame: int, x: int, y: int, visible: list,
                             entries: List[Entry]):
        """The CRT sweep follows the end of every reply or thinking block still revealing; a
        finished one only lets its trail fade. No other kind animates."""
        # Content always starts two columns in, whether or not the row carries a bullet.
        text_by_row = {y + i: cells(line.prefix.ljust(len(BULLET)) + "".join(s.text for s in line.spans))
                       for i, line in enumerate(visible)}

        def char_at(cx: int, cy: int) -> str:
            row = text_by_row.get(cy, [])
            idx = cx - x
            return row[idx] if 0 <= idx < len(row) else " "

        last_row = {}
        for i, line in enumerate(visible):
            if line.entry is not None and line.spans:
                last_row[id(line.entry)] = (i, display_width("".join(s.text for s in line.spans)))
        for entry in entries:
            if not entry.body.animated:
                continue
            revealing = not entry.body.done or not entry.final
            at = None
            if revealing and self.scroll_offset == 0 and id(entry) in last_row:
                row, length = last_row[id(entry)]
                at = (x + 2 + length, y + row)
            fg = self.COLOR_THINKING if entry.kind == KIND_THINKING else self.COLOR_ASSISTANT
            entry.body.draw_cursor(screen, frame, at, char_at=char_at, fg=fg, row_start_x=x + 2)

    # status, rules, composer, footer ---------------------------------------------------------

    def _draw_status(self, screen: TUIScreen, x: int, y: int, width: int):
        screen.reset_color()
        screen.write_at(x, y, " " * width)
        spans = parse_html(self._status or "")
        text = "".join(s.text for s in spans).strip()
        screen.set_color(self.COLOR_DIM)
        screen.italic()
        screen.write_at(x, y, clip(text, width))
        screen.reset_color()

    def _draw_rule(self, screen: TUIScreen, x: int, y: int, width: int):
        screen.reset_color()
        screen.set_color(self.COLOR_RULE)
        screen.write_at(x, y, "─" * width)
        screen.reset_color()

    def _paragraph_rows(self, text_w: int) -> List[List[str]]:
        """The composer text as display rows, one list per paragraph (Shift+Enter starts a
        new one), each paragraph wrapped at `text_w` cells."""
        paragraphs = self.editor.text.split("\n")
        out = []
        for i, paragraph in enumerate(paragraphs):
            rows = chunks(paragraph, text_w)
            if i == len(paragraphs) - 1 and display_width(rows[-1]) >= text_w:
                rows.append("")   # the cursor sits at the start of a fresh row
            out.append(rows)
        return out

    def _input_rows(self, text_w: int) -> List[str]:
        return [row for rows in self._paragraph_rows(text_w) for row in rows]

    def _staged_rows(self, spare: int) -> int:
        """The rows the staged images take above the composer text: thumbnails with their
        chips under them, or the chips alone on one row when thumbnails would leave the
        transcript fewer than `MIN_TRANSCRIPT_ROWS` of the `spare` rows (the gap included)."""
        if not self._images or self._pending_secret is not None:
            return 0
        if spare - (self.THUMBNAIL_ROWS + 1) - 1 >= self.MIN_TRANSCRIPT_ROWS:
            return self.THUMBNAIL_ROWS + 1
        trace(f"ChatPane: {spare} spare rows are too few for thumbnails, the staged images show as chips")
        return 1

    def _draw_staged(self, screen: TUIScreen, x: int, y: int, width: int, rows: int):
        """The staged images under the prompt's indent: `rows` - 1 rows of thumbnails side by
        side, each with its chip under it (on one row, the chips alone); what does not fit
        the width is clipped, a thumbnail that does not fit is left out."""
        screen.reset_color()
        for r in range(rows):
            screen.write_at(x, y + r, " " * width)
        thumb_rows = rows - 1
        left, right = x + len(PROMPT), x + width
        for i, image in enumerate(self._images):
            chip = image.chip(i + 1)
            if thumb_rows:
                cols, thumb_rows_fitted = image.picture.fit(right - left, thumb_rows)
                if left + cols > right:
                    trace(f"ChatPane: thumbnail {i + 1} of {len(self._images)} does not fit the composer's width")
                    break
                screen.image(left, y, image.picture, cols, thumb_rows_fitted)
                slot = max(cols, len(chip))
            else:
                slot = len(chip)
            screen.set_color(self.COLOR_CHIP)
            screen.write_at(left, y + thumb_rows, clip(chip, max(0, right - left)))
            screen.reset_color()
            left += slot + self.THUMBNAIL_GAP
            if left >= right:
                break

    def _draw_composer(self, screen: TUIScreen, frame: int, x: int, y: int, width: int,
                       rows: int, all_rows: List[str], text_w: int):
        for r in range(rows):
            screen.reset_color()
            screen.write_at(x, y + r, " " * width)

        cursor_row, cursor_col = self._cursor_position(text_w)
        # A composer longer than MAX_INPUT_ROWS shows the window of rows ending at the cursor.
        first_row = max(0, cursor_row - rows + 1) if len(all_rows) > rows else 0
        shown = all_rows[first_row:first_row + rows]

        screen.set_color(self.COLOR_PROMPT)
        screen.bold()
        screen.write_at(x, y, PROMPT)
        text_x = x + len(PROMPT)

        if self._pending_secret is not None:
            self._draw_secret_field(screen, frame, text_x, y, text_w)
            return
        if self._pending_approval is not None:
            screen.reset_color()
            screen.set_color(self.COLOR_APPROVAL)
            screen.bold()
            screen.write_at(text_x, y, self.APPROVAL_HINT[:text_w])
            self._input_cursor.hide()
            self._input_cursor.draw(screen, frame)
            return

        screen.reset_color()
        screen.set_color(self.COLOR_INPUT)
        for r, row_text in enumerate(shown):
            if row_text:
                screen.write_at(text_x, y + r, row_text)

        visible_row = cursor_row - first_row
        cy = y + max(0, min(rows - 1, visible_row))
        cx = text_x + cursor_col
        shown_cells = [cells(row_text) for row_text in shown]
        row_cells = shown_cells[visible_row] if 0 <= visible_row < len(shown) else []
        char_at_cursor = row_cells[cursor_col] if cursor_col < len(row_cells) else " "

        def char_at(px: int, py: int) -> str:
            r = py - y
            if 0 <= r < len(shown_cells):
                idx = px - text_x
                if 0 <= idx < len(shown_cells[r]):
                    return shown_cells[r][idx]
            return " "

        self._input_cursor.update(cx, cy)
        self._input_cursor.draw(screen, frame, char_at_cursor=char_at_cursor, char_at=char_at,
                                fg=self.COLOR_INPUT)

    def _cursor_position(self, text_w: int) -> tuple:
        """The composer cursor as (row, cell column) over the rows `_input_rows` made — the
        cursor index counts characters, the column counts cells. The newline ending a
        paragraph is a character of the text but of no row."""
        before = self.editor.text[:self.editor.cursor]
        paragraphs = self._paragraph_rows(text_w)
        rows = paragraphs[before.count("\n")]
        first_row = sum(len(p) for p in paragraphs[:before.count("\n")])
        index = len(before) - (before.rfind("\n") + 1)
        for r, row_text in enumerate(rows):
            if index <= len(row_text) and (r == len(rows) - 1 or index < len(row_text)):
                return first_row + r, column(row_text, index)
            index -= len(row_text)
        return first_row + len(rows) - 1, display_width(rows[-1])

    def _draw_secret_field(self, screen: TUIScreen, frame: int, x: int, y: int, text_w: int):
        """`<title>: ` then the value as dots (or as typed when revealed), on one row, the
        local-only note dim at the right edge while the field keeps enough room; a value
        longer than the room scrolls so the cursor stays in view."""
        editor = self._secret_editor
        title = self._pending_secret.body.text.split("\n", 1)[0]
        label = f"{title}: "
        room = text_w - display_width(label)
        if room < 8:
            label = "secret: "
            room = max(1, text_w - len(label))
        note_w = display_width(self.SECRET_NOTE) + 2
        note = room - note_w >= self.SECRET_NOTE_MIN_FIELD
        if note:
            room -= note_w
        else:
            trace(f"ChatPane: secret field too narrow ({room} cells) for the local-only note")
        shown = editor.text if self._secret_revealed else self.MASK * len(editor.text)
        start = 0
        while start < editor.cursor and column(shown, editor.cursor) - column(shown, start) >= room:
            start += 1
        visible = clip(shown[start:], room)

        screen.reset_color()
        screen.set_color(self.COLOR_APPROVAL)
        screen.bold()
        screen.write_at(x, y, label)
        field_x = x + display_width(label)
        screen.reset_color()
        screen.set_color(self.COLOR_INPUT)
        if visible:
            screen.write_at(field_x, y, visible)
        if note:
            note_x = x + text_w - display_width(self.SECRET_NOTE)
            screen.reset_color()
            screen.set_color(self.COLOR_APPROVAL)
            screen.bold()
            screen.write_at(note_x, y, self.SECRET_NOTE_LEAD)
            screen.reset_color()
            screen.set_color(self.COLOR_DIM)
            screen.write_at(note_x + display_width(self.SECRET_NOTE_LEAD), y, self.SECRET_NOTE_REST)
        cx = field_x + column(shown, editor.cursor) - column(shown, start)
        row_cells = cells(label + visible)

        def char_at(px: int, py: int) -> str:
            # The cursor's trail repaints the cells it swept over — they must keep their text.
            idx = px - x
            return row_cells[idx] if py == y and 0 <= idx < len(row_cells) else " "

        self._input_cursor.update(cx, y)
        self._input_cursor.draw(screen, frame, char_at_cursor=char_at(cx, y), char_at=char_at, fg=self.COLOR_INPUT)

    def _draw_footer(self, screen: TUIScreen, x: int, y: int, width: int):
        """Key hints at the left; at the right the notice (red, bold) or else the feed strip;
        between them, centred, the progress bar when `footer_progress` is set (the first
        curation) and there is room for it. A narrow pane clips the hints first — the strip
        is the news, the hints are known — and a strip wider than the pane loses its last
        chips."""
        screen.reset_color()
        screen.write_at(x, y, " " * width)
        if self._pending_secret is not None:
            footer = self.SECRET_FOOTER
        else:
            footer = f"{self.FOOTER}   {self.DEBUG_HINT} {'on' if self.show_debug else 'off'}"
        notice = clip(self.footer_notice, width) if self.footer_notice else None
        chips = [] if notice else list(self.footer_chips)
        while chips and strip_width(chips) > width:
            chips.pop()
        right_w = display_width(notice) if notice else strip_width(chips)
        hints_w = width - right_w - 2 if right_w else width
        if self._visible_links:
            # A link on screen: in a terminal without hyperlinks a wrapped link is a dead
            # click, so the key that opens it leads the hints, bright; in one with them, a
            # dim word on how to click.
            if screen.hyperlinks:
                link_hint = clip(f" {self.CLICK_HINT}", max(0, hints_w))
                screen.set_color(self.COLOR_DIM)
            else:
                link_hint = clip(f" {self.LINK_HINT}", max(0, hints_w))
                screen.set_color(self.COLOR_LINK_HINT)
                screen.bold()
            screen.write_at(x, y, link_hint)
            screen.reset_color()
            x_hints, hints_w = x + display_width(link_hint) + 1, hints_w - display_width(link_hint) - 1
        else:
            x_hints = x
        hints = clip(footer, max(0, hints_w))
        screen.set_color(self.COLOR_DIM)
        screen.write_at(x_hints, y, hints)
        if self.footer_progress is not None:
            self._draw_progress(screen, x_hints + display_width(hints), x + width - right_w, y)
        if notice:
            screen.set_color(self.COLOR_NOTICE_FG)
            screen.bold()
            screen.write_at(x + width - right_w, y, notice)
        elif chips:
            self._draw_chips(screen, chips, x + width - right_w, y)
        screen.reset_color()

    def _draw_progress(self, screen: TUIScreen, left: int, right: int, y: int):
        """Label, bar and ETA centred in the gap left..right. A gap too narrow for all three
        loses the label, then the ETA, then shows nothing."""
        room = right - left - 2 * self.PROGRESS_BAR_AIR
        label = self.PROGRESS_LABEL
        eta = self.PROGRESS_ETA.format(format_eta(self.footer_eta)) if self.footer_eta is not None else ""
        for label, eta in ((label, eta), ("", eta), ("", "")):
            total = display_width(label) + self.PROGRESS_BAR_WIDTH + display_width(eta)
            if room >= total:
                break
        else:
            return
        start = left + (right - left - total) // 2
        screen.set_color(self.COLOR_DIM)
        screen.write_at(start, y, label)
        bar_x = start + display_width(label)
        screen.draw_progress_bar(bar_x, y, self.PROGRESS_BAR_WIDTH, self.footer_progress,
                                 Theme.COLOR_YELLOW, Theme.COLOR_YELLOW_DIM)
        screen.set_color(self.COLOR_DIM)
        screen.write_at(bar_x + self.PROGRESS_BAR_WIDTH, y, eta)
        screen.reset_color()

    def _draw_chips(self, screen: TUIScreen, chips: List[FeedChip], x: int, y: int):
        screen.set_color(self.COLOR_DIM)
        screen.write_at(x, y, CHIP_HEADER)
        x += display_width(CHIP_HEADER)
        for i, chip in enumerate(chips):
            if i:
                screen.set_color(self.COLOR_DIM)
                screen.write_at(x, y, CHIP_SEPARATOR)
                x += display_width(CHIP_SEPARATOR)
            screen.set_color(chip.color)
            if chip.background is not None:
                screen.set_bg_color(chip.background)
            screen.write_at(x, y, chip.symbol)
            x += display_width(chip.symbol) + 1
            screen.reset_color()
            screen.set_color(self.COLOR_DIM)
            screen.write_at(x, y, chip.label)
            x += display_width(chip.label)


# The hub's outcomes and re-ask details lead with a marker meant for Telegram (✅, ❌, ⚠, ⏱);
# the console's bullet already carries the line, so it goes without.
_OUTCOME_MARKER = re.compile(r"^[^\w\s(]+\s*")

SHIMMER_SPEED = 0.11        # radians per frame (~1 sweep per 1.9 s at 30 FPS)
SHIMMER_SPACING = 0.55      # radians per character


def shimmer(base: Color, frame: int, index: int) -> Color:
    """`base` scaled by a sum of three sines over animation phase `a` and character
    position `b`: a main wave travelling along the text plus two slower, offset ripples.
    The scale stays within [0.50, 1.00] so the text is never lost nor blown out."""
    a = frame * SHIMMER_SPEED
    b = index * SHIMMER_SPACING
    wave = 0.40 * math.sin(a + b) + 0.40 * math.sin(-a + 1.5 * b) + 0.20 * math.sin(a*0.7 * b*2)
    k = 0.75 + 0.25 * wave
    return Color(*(min(255, int(round(c * k))) for c in (base.r, base.g, base.b)))


def _header_spans(header: str) -> List[StyledSpan]:
    """A header is "Name" or "Name (detail)"; the detail gets its own, darker span."""
    name, sep, detail = header.partition(" (")
    spans = [StyledSpan(text=name, italic=True)]
    if sep:
        spans.append(StyledSpan(text=sep + detail, italic=True))
    return spans


def _lstrip_bullet(spans: List[StyledSpan]) -> List[StyledSpan]:
    """Drop leading whitespace and a leading bullet from parsed notification HTML."""
    out = list(spans)
    while out:
        text = out[0].text.lstrip()
        if text.startswith("•"):
            text = text[1:].lstrip()
        if text:
            out[0] = replace(out[0], text=text)
            break
        out.pop(0)
    return out
