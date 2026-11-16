import textwrap
import time
from datetime import timedelta
from typing import Dict, FrozenSet, List, NamedTuple, Optional, Tuple

from assistant.integration.Accounts import render_accounts_status, STATE_OK
from assistant.integration.Feed import Feed
from assistant.integration.Integration import Integration, IntegrationState
from assistant.integration.IntegrationMonitor import IntegrationMonitor, MacOnlyIntegration
from common.logging import trace
from interfaces.console import Theme
from interfaces.console.framework.TUIScreen import TUIScreen, Color
from interfaces.console.tabs.TabView import TabView, SUB_CONTENT_TOP
from interfaces.console.tabs.WatchListTab import span
from interfaces.console.widgets import Board, Tile
from interfaces.console.widgets.Board import Fact, SummaryTile, Table

NAME_W, STATUS_W = 22, 14
DESCRIPTION_MIN_W = 24                   # a narrower table drops the description
STACKED_DETAIL_H = Tile.CARD_CHROME + 6   # the description, three rows of details, a blank one, the status
STACKED_TABLE_MIN_ROWS = 4               # a screen too short for these and the detail shows the table alone
DETAILS_MAX_ROWS = 6
ERROR_MAX_ROWS = 3
STARTING = "integrations are still starting"
ALL_ENABLED = "everything is enabled"
HINT = "↑↓ select  To turn one on or off, ask Keep in the Home chat"


class Status(NamedTuple):
    """How a state reads in the catalog; `attention` is an enabled integration that is not working."""
    label: str
    color: Color
    attention: bool = False


STATUS = {
    IntegrationState.OK: Status("enabled", Theme.COLOR_GREEN),
    IntegrationState.UNKNOWN: Status("starting", Theme.COLOR_CYAN),
    IntegrationState.WARNING: Status("check failed", Theme.COLOR_YELLOW, True),
    IntegrationState.ERROR: Status("error", Theme.COLOR_RED, True),
    IntegrationState.NOT_CONFIGURED: Status("not configured", Theme.COLOR_YELLOW, True),
    IntegrationState.NEEDS_2FA: Status("needs 2FA", Theme.COLOR_PURPLE, True),
    IntegrationState.RATE_LIMITED: Status("rate limited", Theme.COLOR_ORANGE, True),
    IntegrationState.DISABLED: Status("disabled", Theme.COLOR_DIM),
}
# An integration that needs macOS, on a server that is not a Mac: disabled, and not to be turned on.
NEEDS_MAC = Status("needs macOS", Theme.COLOR_DIM)


class Entry(NamedTuple):
    """One integration of the catalog: the registered object (a placeholder for a disabled or
    unbuilt one) and the class behind it."""
    integration: Integration
    cls: type

    @property
    def name(self) -> str:
        return self.integration.name

    @property
    def enabled(self) -> bool:
        return self.integration.state is not IntegrationState.DISABLED

    @property
    def needs_mac(self) -> bool:
        return isinstance(self.integration, MacOnlyIntegration)

    @property
    def status(self) -> Status:
        return NEEDS_MAC if self.needs_mac else STATUS[self.integration.state]


class CatalogTab(TabView):
    """The first Integrations sub-tab: every integration there is, the enabled ones first, with what each one is.

    A board in the Dashboard's tiles: summary tiles (how many are enabled, which of those need
    attention, how many are disabled), a table of them all — enabled in their health colour,
    disabled dimmed under them — and the selected one in full beside it (under it on a narrow
    screen): what it is and what it does for the user (`Integration.description` and
    `details`, the first two paragraphs of its docstring), its health, the config keys it
    misses, what it brings (a feed, private data, a tab of its own) and the words that turn it
    on or off. Read from the IntegrationMonitor, placeholders included, so it lists the
    integrations without a tab of their own too (and the voice engines and the Telegram bot
    when their interface registers them). Read-only: enabling and disabling are the
    assistant's `enable_integration` / `disable_integration`."""

    def __init__(self, tab_integrations: FrozenSet[type] = frozenset()):
        """`tab_integrations`: the integration classes that bring a sub-tab while enabled."""
        self._tab_integrations = tab_integrations
        self._cards: List[Tile.Card] = []
        self._order: List[str] = []
        self._selected: Optional[str] = None
        self._top = 0
        self._room = 1
        self._listed: Optional[Tuple[Tuple[str, bool], ...]] = None
        self._texts: Dict[type, Tuple[str, str]] = {}    # class -> its description and details

    @property
    def name(self) -> str:
        return "Catalog"

    def handle_input(self, key: str) -> bool:
        step = {'up': -1, 'down': 1, 'pageup': -self._room, 'pagedown': self._room}.get(key)
        if step is None:
            return False
        if self._order:
            index = self._order.index(self._selected) if self._selected in self._order else 0
            self._selected = self._order[max(0, min(len(self._order) - 1, index + step))]
        return True

    def _entries(self) -> List[Entry]:
        """Every registered integration, the enabled ones first, each group by name."""
        entries = sorted((Entry(i, getattr(i, "cls", type(i))) for i in IntegrationMonitor.get_instance().of_type(Integration)),
                         key=lambda entry: (not entry.enabled, entry.name.lower()))
        listed = tuple((entry.name, entry.enabled) for entry in entries)
        if listed != self._listed:
            self._listed = listed
            trace(f"Catalog tab: listing {len(entries)} integrations, {sum(1 for e in entries if e.enabled)} enabled")
        return entries

    def _text(self, cls: type) -> Tuple[str, str]:
        if cls not in self._texts:
            self._texts[cls] = (cls.description(), cls.details())
        return self._texts[cls]

    def _tags(self, entry: Entry) -> str:
        tags = []
        if issubclass(entry.cls, Feed):
            tags.append("feed")
        if entry.cls.sensitive:
            tags.append("private data")
        if entry.cls in self._tab_integrations:
            tags.append("own tab")
        return " · ".join(tags)

    def draw(self, screen: TUIScreen, frame: int):
        entries = self._entries()
        self._order = [entry.name for entry in entries]
        if self._selected not in self._order:
            self._selected = self._order[0] if self._order else None

        self._cards = []
        width = screen.width - Board.LEFT - Board.RIGHT_MARGIN
        bottom = screen.height - 1                # the last row is the key hints'
        y = SUB_CONTENT_TOP
        if bottom - y - Board.SUMMARY_H - 1 > Tile.CARD_CHROME + 1:
            Board.draw_summary(screen, self._cards, y, width, self._tiles(entries))
            y += Board.SUMMARY_H + 1
        height = bottom - y
        if not entries:
            Board.draw_notice(screen, self._cards, Board.LEFT, y, width, height, "Integrations", STARTING)
        else:
            index = self._order.index(self._selected)
            if screen.width >= Board.SIDE_BY_SIDE_MIN:
                table_w, detail_w = Board.split(width)
                self._draw_table(screen, Board.LEFT, y, table_w, height, entries, index)
                self._draw_detail(screen, Board.LEFT + table_w + Board.GAP, y, detail_w, height, entries[index])
            else:
                table_h = height - 1 - STACKED_DETAIL_H
                if table_h < Tile.CARD_CHROME + 1 + STACKED_TABLE_MIN_ROWS:
                    table_h = height
                table_h = min(table_h, len(entries) + Tile.CARD_CHROME + 2)   # the rows, their titles and the gap
                self._draw_table(screen, Board.LEFT, y, width, table_h, entries, index)
                self._draw_detail(screen, Board.LEFT, y + table_h + 1, width, height - table_h - 1, entries[index])
        Tile.tint_cards(screen, self._cards)
        Board.draw_hints(screen, HINT + self._ask(entries[self._order.index(self._selected)]) if entries else HINT)

    @staticmethod
    def _ask(entry: Entry) -> str:
        if entry.needs_mac:
            return ""       # nothing turns it on here
        return f': "{"disable" if entry.enabled else "enable"} {entry.name}"'

    # --- Summary tiles ---

    @staticmethod
    def _tiles(entries: List[Entry]) -> List[SummaryTile]:
        if not entries:
            return [("Integrations", "–", Theme.COLOR_DIM, [(STARTING, Theme.COLOR_DIM)])]
        enabled = [entry for entry in entries if entry.enabled]
        attention = [entry for entry in enabled if entry.status.attention]
        disabled = len(entries) - len(enabled)
        if attention:
            attention_tile = ("Needs attention", str(len(attention)), attention[0].status.color,
                              [(f"{entry.name} {entry.status.label}", entry.status.color) for entry in attention])
        elif enabled:
            attention_tile = ("Needs attention", "all clear", Theme.COLOR_GREEN,
                              [("every enabled integration is working", Theme.COLOR_DIM)])
        else:
            attention_tile = ("Needs attention", "–", Theme.COLOR_DIM, [("nothing is enabled", Theme.COLOR_DIM)])
        return [("Enabled", str(len(enabled)), Theme.COLOR_GREEN if enabled else Theme.COLOR_DIM,
                 [(f"{len(enabled)} of {len(entries)} integrations", Theme.COLOR_LABEL)]),
                attention_tile,
                ("Disabled", str(disabled), Theme.COLOR_BRIGHT if disabled else Theme.COLOR_DIM,
                 [(f"{disabled} ready to be turned on", Theme.COLOR_LABEL) if disabled else (ALL_ENABLED, Theme.COLOR_DIM)])]

    # --- The table ---

    @staticmethod
    def _columns(inner_w: int) -> List[Tuple[str, Optional[int]]]:
        """The columns a table `inner_w` wide has room for; the last one takes what is left."""
        if inner_w - 2 - (NAME_W + 2) - (STATUS_W + 2) < DESCRIPTION_MIN_W:
            return [("Integration", NAME_W), ("Status", None)]
        return [("Integration", NAME_W), ("Status", STATUS_W), ("Description", None)]

    def _cells(self, entry: Entry, chosen: bool) -> Dict[str, Board.Cell]:
        if chosen:
            name_color = Theme.COLOR_BRIGHT
        else:
            name_color = Theme.COLOR_TEXT if entry.enabled else Theme.COLOR_LABEL
        return {"Integration": (entry.name, name_color, chosen),
                "Status": (entry.status.label, entry.status.color),
                "Description": (self._text(entry.cls)[0].removesuffix("."),
                               Theme.COLOR_LABEL if entry.enabled else Theme.COLOR_DIM)}

    def _draw_table(self, screen: TUIScreen, x: int, y: int, width: int, height: int, entries: List[Entry], selected: int):
        columns = self._columns(width - 4)
        enabled = sum(1 for entry in entries if entry.enabled)
        gap = 0 < enabled < len(entries)        # a blank row parts the enabled from the disabled
        rows = []
        for i, entry in enumerate(entries):
            if gap and i == enabled:
                rows.append((Theme.COLOR_FAINT, []))
            cells = self._cells(entry, i == selected)
            rows.append((entry.status.color if entry.enabled else Theme.COLOR_FAINT, [cells[title] for title, _ in columns]))
        table = Table("Integrations", columns, rows, STARTING, f"{enabled} of {len(entries)} enabled")
        self._room = max(1, height - Tile.CARD_CHROME - 1)
        self._top = Board.draw_table(screen, self._cards, x, y, width, height, table, self._top,
                                     selected + (gap and selected >= enabled))

    # --- The selected integration ---

    @staticmethod
    def _wrapped(label: str, text: str, width: int, color: Color, max_rows: int) -> List[Fact]:
        """`text` as a fact that runs over as many rows as it needs, up to `max_rows`."""
        lines = textwrap.wrap(text, max(8, width), max_lines=max_rows, placeholder="…") or [""]
        return [Fact(label if i == 0 else "", line, color) for i, line in enumerate(lines)]

    def _facts(self, entry: Entry, value_w: int) -> List[Fact]:
        integration, status = entry.integration, entry.status
        facts = [Fact("Status", status.label, status.color, bold=True)]
        if entry.enabled:
            checked = integration.last_check_time
            facts.append(Fact("Last check", f"{span(timedelta(seconds=time.time() - checked))} ago", Theme.COLOR_TEXT)
                         if checked else Fact("Last check", "not yet", Theme.COLOR_DIM))
            if integration.last_error:
                facts += self._wrapped("Problem", integration.last_error, value_w, status.color, ERROR_MAX_ROWS)
            accounts = integration.accounts_status()
            if accounts:
                color = Theme.COLOR_TEXT if all(a["state"] == STATE_OK for a in accounts) else status.color
                facts += self._wrapped("Accounts", render_accounts_status(accounts), value_w, color, ERROR_MAX_ROWS)
            missing = integration.missing_config_keys()
            if missing:
                facts += self._wrapped("Missing", ", ".join(missing), value_w, Theme.COLOR_YELLOW, ERROR_MAX_ROWS)
            elif integration.config_keys:
                count = len(integration.config_keys)
                facts.append(Fact("Config", "its key is set" if count == 1 else f"all {count} keys are set", Theme.COLOR_LABEL))
        tags = self._tags(entry)
        facts.append(Fact("Brings", tags, Theme.COLOR_TEAL) if tags else Fact("Brings", "–", Theme.COLOR_FAINT))
        if entry.needs_mac:
            return facts + self._wrapped("Turn on", integration.last_error, value_w, Theme.COLOR_LABEL, ERROR_MAX_ROWS)
        facts.append(Fact("Turn off" if entry.enabled else "Turn on", f"ask Keep{self._ask(entry)}", Theme.COLOR_TEXT))
        return facts

    def _draw_detail(self, screen: TUIScreen, x: int, y: int, width: int, height: int, entry: Entry):
        """The selected integration: what it is, then how it stands and what turns it on or off."""
        if height < Tile.CARD_CHROME + 1:
            return
        Tile.draw_card(screen, self._cards, x, y, width, height, entry.name, entry.status.label, entry.status.color)
        inner_w = width - 4
        room = height - Tile.CARD_CHROME
        description, details = self._text(entry.cls)
        text_room = max(1, room - 2)                # the status keeps its row on a short tile
        lines = [(line, Theme.COLOR_BRIGHT) for line in textwrap.wrap(description, max(8, inner_w))][:text_room]
        budget = min(DETAILS_MAX_ROWS, text_room - len(lines))
        more = [(line, Theme.COLOR_LABEL) for line in
                textwrap.wrap(details, max(8, inner_w), max_lines=budget, placeholder="…")] if budget > 0 else []
        if lines and more and len(lines) + 1 + len(more) <= text_room:
            lines.append(("", Theme.COLOR_LABEL))
        lines += more
        for i, (line, color) in enumerate(lines[:room]):
            screen.set_color(color)
            screen.write_at(x + 2, y + Tile.CARD_TOP + i, line)
        screen.reset_color()
        first_row = Tile.CARD_TOP + (len(lines) + 1 if lines else 0)
        Board.draw_facts(screen, x, y, width, height, self._facts(entry, inner_w - Board.LABEL_W), first_row=first_row)
