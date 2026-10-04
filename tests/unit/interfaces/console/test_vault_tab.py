"""VaultTab: the secrets drawn masked in the Dashboard's tiles (summary, table, the form tile,
the help), and the add / edit / delete / reveal flows driven through handle_input with the key
names TUIFloxyConsole._parse_key produces (Enter is '\\r', Escape 'escape', Tab '\\t')."""
import re

import pytest

from common.vault.Redactor import Redactor
from common.vault.Vault import Vault
from interfaces.console import Theme, TUIState
from interfaces.console.framework.TUIScreen import TUIScreen
from interfaces.console.tabs.VaultTab import HELP, HELP_MAX_WIDTH, HELP_TITLE, VaultTab
from interfaces.console.widgets.Board import COLOR_PROMPT, LABEL_W as FORM_LABEL_W, LEFT, SIDE_BY_SIDE_MIN
from interfaces.console.widgets import Markdown

# 0-based string indexes of what a tile at the left margin draws: its content, and a table cell
CONTENT = LEFT + 1
ID = CONTENT + 2

PHONE = "+46 70-174 06 05"
EMAIL = "m@mlab.test"
PLATE = "KEP 47L"
RAW_VALUES = (PHONE, "0701740605", "46701740605", EMAIL, PLATE, "KEP47L")
ENTER = '\r'
TALL = 60        # rows enough for the summary tiles and the whole help under the table


@pytest.fixture
def vault(tmp_path):
    v = Vault(tmp_path / "vault.bin", "test")
    v.add("phone", "Marcus's mobile", PHONE)       # id marcus-s-mobile
    v.add("email", "Work e-mail", EMAIL)           # id work-e-mail
    v.add("plate", "The Model Y", PLATE)           # id the-model-y
    return v


@pytest.fixture
def tab(vault):
    return VaultTab(vault)


def rows(screen: TUIScreen) -> list[str]:
    return ["".join(c.char for c in line).rstrip() for line in screen.next_frame]


def draw(tab: VaultTab, width=110, height=24) -> TUIScreen:
    screen = TUIScreen(width, height)
    tab.draw(screen, 0)
    return screen


def lines(tab: VaultTab, **kwargs) -> list[str]:
    return rows(draw(tab, **kwargs))


def press(tab: VaultTab, *keys: str):
    for key in keys:
        assert tab.handle_input(key)


def type_text(tab: VaultTab, text: str):
    press(tab, *text)


def style_at(screen: TUIScreen, row: int, col: int):
    """The style of the cell at 0-based (row, col) — the indexes of rows()."""
    return screen.next_frame[row][col].style


def row_with(lines_: list[str], needle: str) -> str:
    matches = [l for l in lines_ if needle in l]
    assert len(matches) == 1, (needle, matches)
    return matches[0]


def list_rows(lines_: list[str]) -> list[str]:
    """The table's rows, each from its id on."""
    return [l[ID:] for l in lines_ if l[CONTENT:CONTENT + 2] == "● "]


def list_ids(lines_: list[str]) -> list[str]:
    return [row.split()[0] for row in list_rows(lines_)]


def list_row(lines_: list[str], secret_id: str) -> str:
    matches = [row for row in list_rows(lines_) if row.startswith(secret_id + " ")]
    assert len(matches) == 1, (secret_id, matches)
    return matches[0]


def table_row(lines_: list[str], secret_id: str) -> int:
    """The 0-based screen row of `secret_id` in the table."""
    matches = [i for i, l in enumerate(lines_) if l[CONTENT:].startswith(f"● {secret_id} ")]
    assert len(matches) == 1, (secret_id, matches)
    return matches[0]


def selected_id(screen: TUIScreen) -> str:
    """The id of the one table row that stands on the selection's background."""
    out = rows(screen)
    chosen = [l[ID:].split()[0] for i, l in enumerate(out)
              if l[CONTENT:CONTENT + 2] == "● " and style_at(screen, i, ID).bg == Theme.COLOR_SELECTED_BG]
    assert len(chosen) == 1, chosen
    return chosen[0]


def field(lines_: list[str], label: str) -> str:
    """The value of the form tile's `label` row (a stacked layout: the form at the left margin)."""
    matches = [l for l in lines_ if l[CONTENT:CONTENT + FORM_LABEL_W].rstrip() == label]
    assert len(matches) == 1, (label, matches)
    return matches[0][CONTENT + FORM_LABEL_W:]


def waiting(screen: TUIScreen) -> list[str]:
    """The labels of the form rows drawn as the step the flow waits on."""
    return [l[CONTENT:CONTENT + FORM_LABEL_W].strip() for i, l in enumerate(rows(screen))
            if len(l) > CONTENT and l[CONTENT] != " " and style_at(screen, i, CONTENT).fg == COLOR_PROMPT]


def help_rows(screen: TUIScreen) -> list[str]:
    """The help tile's text rows (a stacked layout): from under its title strip to the hint row."""
    out = rows(screen)
    title = out.index(row_with(out, HELP_TITLE))
    return [l[CONTENT:] for l in out[title + 2:screen.height - 1]]


def opening(paragraph: str) -> str:
    """The first words of a HELP paragraph as they read once laid out."""
    return " ".join(re.sub(r"[*>]", "", paragraph).split()[:3])


@pytest.mark.unit
class TestTable:
    def test_the_tile_counts_the_secrets_and_names_the_file(self, tab):
        title = row_with(lines(tab), "vault.bin")
        assert "3" in title.split()

    def test_rows_sorted_by_id_under_their_column_titles_with_masked_values(self, tab):
        out = lines(tab)
        assert list_ids(out) == ["marcus-s-mobile", "the-model-y", "work-e-mail"]
        first, second, third = list_rows(out)
        assert "phone" in first and "Marcus's mobile" in first and first.endswith("•••• 0605")
        assert "plate" in second and second.endswith("KEP •••")
        assert "email" in third and third.endswith("m•••@mlab.test")
        top = table_row(out, "marcus-s-mobile")
        titles = out[top - 1]
        assert titles[ID:].split() == ["Id", "Kind", "Description", "Value"]
        assert titles.index("Kind") == out[top].index("phone")
        assert titles.index("Value") == out[top].index("••••")

    def test_the_selected_row_is_a_bold_id_on_the_selection_bar_and_the_titles_are_dim(self, tab):
        screen = draw(tab)
        out = rows(screen)
        top = table_row(out, "marcus-s-mobile")
        assert style_at(screen, top - 1, ID).fg == Theme.COLOR_DIM and not style_at(screen, top - 1, ID).bold
        first, second = style_at(screen, top, ID), style_at(screen, top + 1, ID)
        assert first.bg == Theme.COLOR_SELECTED_BG and first.bold
        assert style_at(screen, top, len(out[top]) + 5).bg == Theme.COLOR_SELECTED_BG    # past the text too
        assert second.bg == Theme.COLOR_PANEL_BG and not second.bold

    def test_a_revealed_value_is_bold_in_the_secret_orange(self, tab):
        press(tab, 'v')
        screen = draw(tab)
        out = rows(screen)
        top = table_row(out, "marcus-s-mobile")
        column = out[top].index(PHONE)
        assert style_at(screen, top, column).fg == Theme.COLOR_SECRET
        assert style_at(screen, top, column).bold
        press(tab, 'v')
        assert style_at(draw(tab), top, column).fg != Theme.COLOR_SECRET

    def test_no_raw_value_appears_anywhere_on_screen(self, tab):
        for width, height in ((110, 24), (110, 48), (SIDE_BY_SIDE_MIN + 20, 48)):
            frame = "\n".join(lines(tab, width=width, height=height))
            for raw in RAW_VALUES:
                assert raw not in frame

    def test_empty_vault_says_so(self, tmp_path):
        tab = VaultTab(Vault(tmp_path / "vault.bin", "test"))
        out = lines(tab)
        assert list_rows(out) == []
        assert "0" in row_with(out, "vault.bin").split()
        assert any("no secrets yet" in l.lower() for l in out)

    def test_a_narrow_or_tiny_screen_draws_without_raising(self, tab):
        for width, height in ((20, 24), (110, 8), (SIDE_BY_SIDE_MIN + 50, 12), (3, 3), (1, 1)):
            out = lines(tab, width=width, height=height)
            assert all(len(l) <= width for l in out)
        press(tab, 'a', ENTER)
        type_text(tab, "x")
        press(tab, ENTER)
        type_text(tab, "value")
        lines(tab, width=12, height=10)

    def test_long_rows_are_cut_at_the_screen_edge(self, tab):
        out = lines(tab, width=40)
        assert all(len(l) <= 40 for l in out)
        assert list_ids(out)[0] == "marcus-s-mobile"

    def test_a_long_list_scrolls_to_keep_the_selected_row_in_view(self, vault):
        for i in range(40):
            vault.add("text", f"Secret {i:02d}", f"value-{i:04d}")
        tab = VaultTab(vault)
        assert "work-e-mail" not in list_ids(lines(tab, height=30))
        press(tab, *['down'] * 50)
        screen = draw(tab, height=30)
        assert selected_id(screen) == "work-e-mail"           # the last id, past the first page
        assert "marcus-s-mobile" not in list_ids(rows(screen))


@pytest.mark.unit
class TestLayout:
    def test_the_summary_tiles_sit_above_the_table_and_give_way_on_a_short_screen(self, tab):
        tall, short = lines(tab, height=TALL), lines(tab, height=24)

        def first_drawn(lines_):
            return next(i for i, l in enumerate(lines_) if l)

        title = table_row(short, "marcus-s-mobile") - 3       # the table's title strip, a blank row, the column titles
        assert first_drawn(short) == title
        assert first_drawn(tall) < table_row(tall, "marcus-s-mobile") - 3
        assert "Vault" in tall[first_drawn(tall)]
        assert list_ids(short) == list_ids(tall)

    def test_the_secrets_tile_counts_each_way_of_matching(self, tab, vault):
        vault.add("password", "Router admin", "correct-horse-9")
        label, value, _, segments = VaultTab._secrets_tile(tab._rows())
        assert value == "4"
        assert [int(text.split()[0]) for text, _ in segments] == [3, 1]     # by format, verbatim
        vault.delete("router-admin")
        assert len(VaultTab._secrets_tile(tab._rows())[3]) == 1             # a group of none is left out

    def test_the_form_and_the_help_sit_beside_the_table_from_the_side_by_side_width(self, tab):
        placeholder = "{{secret:marcus-s-mobile}}"
        wide = lines(tab, width=SIDE_BY_SIDE_MIN, height=TALL)
        last = table_row(wide, "work-e-mail")
        table_end = wide[last].index("Work e-mail") + len("Work e-mail")
        form = row_with(wide, placeholder)
        assert wide.index(form) <= last and form.index(placeholder) > table_end
        assert row_with(wide, HELP_TITLE).index(HELP_TITLE) > table_end

        stacked = lines(tab, width=SIDE_BY_SIDE_MIN - 1, height=TALL)
        last = table_row(stacked, "work-e-mail")
        form, help_title = stacked.index(row_with(stacked, placeholder)), stacked.index(row_with(stacked, HELP_TITLE))
        assert last < form < help_title
        assert stacked[help_title].index(HELP_TITLE) == CONTENT


@pytest.mark.unit
class TestForm:
    def test_it_shows_the_selected_secret_masked_and_what_the_ai_sees_in_its_place(self, tab, vault):
        out = lines(tab)
        placeholder, _ = Redactor(vault).redact(PHONE)
        assert "Marcus's mobile" in field(out, "Description")
        assert field(out, "Value").strip() == "•••• 0605"
        assert any(placeholder in l for l in out)
        press(tab, 'down')
        out = lines(tab)
        assert field(out, "Value").strip() == "KEP •••"
        assert any(Redactor(vault).redact(PLATE)[0] in l for l in out)

    def test_a_revealed_value_shows_in_the_form_too(self, tab):
        press(tab, 'v')
        assert field(lines(tab), "Value").strip() == PHONE

    def test_view_mode_waits_on_nothing(self, tab):
        assert waiting(draw(tab)) == []

    def test_the_add_flow_marks_the_step_it_waits_on(self, tab):
        press(tab, 'a')
        assert waiting(draw(tab)) == ["Kind"]
        press(tab, ENTER)
        assert waiting(draw(tab)) == ["Description"]
        type_text(tab, "Anna's mobile")
        press(tab, ENTER)
        screen = draw(tab)
        assert waiting(screen) == ["Value"]
        assert "Anna's mobile" in field(rows(screen), "Description")       # the answered step stays on show

    def test_the_delete_form_shows_the_secret_masked(self, tab):
        press(tab, 'd')
        screen = draw(tab)
        out = rows(screen)
        assert field(out, "Value").strip() == "•••• 0605"
        assert len(waiting(screen)) == 1 and waiting(screen)[0] not in ("Description", "Value")
        assert PHONE not in "\n".join(out)


@pytest.mark.unit
class TestHelp:
    """The how-it-works tile: under the table and the form when there is room, gone on a short
    or narrow screen, scrolled with PgUp/PgDn when it does not fit."""

    def test_every_paragraph_is_shown_without_its_markdown_markers(self, tab):
        screen = draw(tab, height=TALL)
        block = help_rows(screen)
        text = "\n".join(block)
        for paragraph in HELP:
            assert opening(paragraph) in text
        assert "**" not in text
        assert all(len(l) <= screen.width for l in rows(screen))

    def test_bold_pieces_are_bright_and_bold_and_the_rest_is_not(self, tab):
        screen = draw(tab, height=TALL)
        out = rows(screen)
        row = out.index(row_with(out, HELP_TITLE)) + 2         # "Secrets are stored encrypted in vault.bin, …"
        plain, bold = out[row].index("Secrets"), out[row].index("vault.bin")
        assert not style_at(screen, row, plain).bold
        assert style_at(screen, row, bold).bold and style_at(screen, row, bold).fg == Markdown.COLOR_BOLD
        assert not style_at(screen, row, bold + len("vault.bin")).bold     # the comma after it

    def test_the_text_never_runs_wider_than_the_cap(self, tab):
        block = help_rows(draw(tab, width=SIDE_BY_SIDE_MIN - 1, height=TALL))
        assert max(len(l) for l in block) <= HELP_MAX_WIDTH
        assert max(len(l) for l in block) > HELP_MAX_WIDTH - 20        # wrapped near the cap, not far under it

    def test_hidden_when_it_would_squeeze_the_list(self, tab):
        out = lines(tab, height=24)
        assert not any(HELP_TITLE in l for l in out)
        assert len(list_rows(out)) == 3

    def test_hidden_on_a_screen_too_narrow_to_read(self, tab):
        out = lines(tab, width=30, height=TALL)
        assert not any(HELP_TITLE in l for l in out)
        assert list_ids(out)[0] == "marcus-s-mobile"

    def test_page_keys_scroll_a_help_that_does_not_fit(self, tab):
        first, last = opening(HELP[0]), opening(HELP[-1])
        height = 40
        text = "\n".join(help_rows(draw(tab, height=height)))
        assert first in text and last not in text
        press(tab, *['pagedown'] * 20)
        text = "\n".join(help_rows(draw(tab, height=height)))
        assert last in text and first not in text
        press(tab, 'pageup')                                   # one page up from the end, not from 20 pages down
        assert "\n".join(help_rows(draw(tab, height=height))) != text
        press(tab, *['pageup'] * 20)
        assert first in "\n".join(help_rows(draw(tab, height=height)))


@pytest.mark.unit
class TestSelectionAndReveal:
    def test_up_and_down_move_the_selection_and_stop_at_the_ends(self, tab):
        press(tab, 'down')
        assert selected_id(draw(tab)) == "the-model-y"
        press(tab, 'down', 'down', 'down')
        assert selected_id(draw(tab)) == "work-e-mail"
        press(tab, 'up', 'up', 'up', 'up')
        assert selected_id(draw(tab)) == "marcus-s-mobile"

    def test_v_reveals_the_selected_row_only_and_toggles_back(self, tab):
        press(tab, 'v')
        out = lines(tab)
        assert list_row(out, "marcus-s-mobile").endswith(PHONE)
        assert list_row(out, "the-model-y").endswith("KEP •••")
        assert list_row(out, "work-e-mail").endswith("m•••@mlab.test")
        press(tab, 'v')
        assert list_row(lines(tab), "marcus-s-mobile").endswith("•••• 0605")

    def test_reveal_follows_the_row_not_the_cursor(self, tab):
        press(tab, 'down', 'v', 'up')
        out = lines(tab)
        assert list_row(out, "marcus-s-mobile").endswith("•••• 0605")
        assert list_row(out, "the-model-y").endswith(PLATE)

    def test_leaving_the_tab_hides_revealed_values(self, tab):
        press(tab, 'v')
        tab.on_deactivate()
        assert PHONE not in "\n".join(lines(tab))

    def test_leaving_the_tab_mid_typing_drops_a_revealed_field(self, tab, vault):
        press(tab, 'a', ENTER)
        type_text(tab, "Anna")
        press(tab, ENTER)
        type_text(tab, "+46 73 123 45 67")
        press(tab, 'tab')                             # reveal the field while typing
        assert "+46 73 123 45 67" in "\n".join(lines(tab))
        tab.on_deactivate()
        assert "+46 73 123 45 67" not in "\n".join(lines(tab))
        assert tab.captures_text is False
        assert len(vault) == 3

    def test_view_keys_it_does_not_know_are_not_consumed(self, tab):
        assert not tab.handle_input('x')
        assert not tab.handle_input('\x18')
        assert not tab.handle_input('escape')


@pytest.mark.unit
class TestAddFlow:
    def test_tab_is_claimed_only_while_the_value_field_is_open(self, tab):
        # The console lets Tab walk the tabs past a text field unless the tab claims it for
        # revealing a masked value; the kind and description steps must not claim it.
        assert not tab.claims_tab
        press(tab, 'a')
        assert not tab.claims_tab
        press(tab, ENTER)
        assert tab.captures_text and not tab.claims_tab
        type_text(tab, "Anna's mobile")
        press(tab, ENTER)
        assert tab.claims_tab
        type_text(tab, "0701234567")
        press(tab, 'tab')                            # the reveal, not a tab switch
        assert "0701234567" in field(lines(tab), "Value") and tab.claims_tab
        press(tab, 'escape')
        assert not tab.claims_tab

    def test_kind_description_value_then_the_row_appears(self, tab, vault):
        press(tab, 'a')
        assert tab.captures_text is False
        assert "phone" in field(lines(tab), "Kind")
        press(tab, 'right')
        assert "card" in field(lines(tab), "Kind")
        press(tab, 'left', 'left')
        assert "text" in field(lines(tab), "Kind")
        press(tab, 'right')                     # wraps back to phone
        press(tab, ENTER)
        assert tab.captures_text is True
        assert waiting(draw(tab)) == ["Description"]
        type_text(tab, "Anna's mobile")
        assert "Anna's mobile" in field(lines(tab), "Description")
        press(tab, ENTER)
        screen = draw(tab)
        assert waiting(screen) == ["Value"] and "phone" in field(rows(screen), "Kind")
        type_text(tab, "+46 73 123 45 67")
        press(tab, ENTER)

        assert tab.captures_text is False
        assert len(vault) == 4
        added = vault.get("anna-s-mobile")
        assert added.kind == "phone" and added.value == "+46 73 123 45 67"
        screen = draw(tab)
        out = rows(screen)
        assert selected_id(screen) == "anna-s-mobile"
        assert list_row(out, "anna-s-mobile").endswith("•••• 4567")
        status = row_with(out, "Added")
        assert "anna-s-mobile" in status
        assert style_at(screen, out.index(status), status.index("Added")).fg == Theme.COLOR_GREEN
        assert "4" in row_with(out, "vault.bin").split()

    def test_the_value_is_masked_while_typing_until_tab_reveals_it(self, tab):
        press(tab, 'a', ENTER)
        type_text(tab, "Anna")
        press(tab, ENTER)
        type_text(tab, "+46 73 123 45 67")
        frame = "\n".join(lines(tab))
        assert "+46 73 123 45 67" not in frame
        assert "•" * len("+46 73 123 45 67") in field(lines(tab), "Value")
        press(tab, 'tab')
        assert "+46 73 123 45 67" in field(lines(tab), "Value")
        press(tab, 'tab')
        assert "+46 73 123 45 67" not in "\n".join(lines(tab))

    def test_letters_v_a_e_d_go_into_the_field_while_typing(self, tab, vault):
        press(tab, 'a', 'right', 'right', 'right', 'right', 'right', 'right', ENTER)   # password
        type_text(tab, "vault dev")
        press(tab, ENTER)
        type_text(tab, "vade-retro-1")
        press(tab, ENTER)
        assert vault.get("vault-dev").value == "vade-retro-1"

    def test_too_short_value_shows_a_red_error_and_creates_nothing(self, tab, vault):
        press(tab, 'a', ENTER)
        type_text(tab, "Door")
        press(tab, ENTER)
        type_text(tab, "12345")
        press(tab, ENTER)
        assert len(vault) == 3
        assert tab.captures_text is True            # still in the value step, the text kept
        screen = draw(tab)
        out = rows(screen)
        error = row_with(out, "too short")
        assert style_at(screen, out.index(error), CONTENT).fg == Theme.COLOR_RED
        assert "•••••" in field(out, "Value") and waiting(screen) == ["Value"]
        press(tab, 'escape')
        assert tab.captures_text is False
        assert not any("too short" in l for l in lines(tab))

    def test_duplicate_value_names_the_existing_secret(self, tab, vault):
        press(tab, 'a', ENTER)
        type_text(tab, "Same phone")
        press(tab, ENTER)
        type_text(tab, "0701740605")
        press(tab, ENTER)
        assert len(vault) == 3
        assert "marcus-s-mobile" in row_with(lines(tab), "Cannot add")

    def test_empty_description_is_refused(self, tab):
        press(tab, 'a', ENTER, ENTER)
        screen = draw(tab)
        out = rows(screen)
        error = row_with(out, "required")
        assert style_at(screen, out.index(error), CONTENT).fg == Theme.COLOR_RED
        assert tab.captures_text is True and waiting(screen) == ["Description"]

    def test_escape_cancels_at_every_step(self, tab, vault):
        press(tab, 'a', 'escape')
        assert tab.captures_text is False and waiting(draw(tab)) == []
        press(tab, 'a', ENTER, 'escape')
        assert tab.captures_text is False
        press(tab, 'a', ENTER)
        type_text(tab, "x")
        press(tab, ENTER, 'escape')
        assert tab.captures_text is False
        assert len(vault) == 3

    def test_captures_text_only_while_a_text_field_is_active(self, tab):
        assert tab.captures_text is False
        press(tab, 'a')
        assert tab.captures_text is False
        press(tab, ENTER)
        assert tab.captures_text is True
        type_text(tab, "x")
        press(tab, ENTER)
        assert tab.captures_text is True
        press(tab, 'escape')
        assert tab.captures_text is False
        press(tab, 'd')
        assert tab.captures_text is False
        press(tab, 'escape', 'e')
        assert tab.captures_text is True

    def test_no_key_of_the_add_flow_touches_tui_state(self, tab, vault, monkeypatch):
        def forbidden(*args, **kwargs):
            raise AssertionError("TUIState must never see a vault field")
        monkeypatch.setattr(TUIState, "load", forbidden)
        monkeypatch.setattr(TUIState, "save", forbidden)
        press(tab, 'a', 'right', 'right', 'right', 'right', 'right', ENTER)   # email
        type_text(tab, "Private e-mail")
        press(tab, ENTER)
        type_text(tab, "marcus@example.com")
        press(tab, 'tab', ENTER)
        assert vault.get("private-e-mail").value == "marcus@example.com"
        lines(tab)


@pytest.mark.unit
class TestEditFlow:
    def test_tab_is_claimed_once_the_edit_reaches_the_value_field(self, tab):
        press(tab, 'e')
        assert tab.captures_text and not tab.claims_tab
        press(tab, ENTER)                            # unchanged description: straight to the value
        assert tab.claims_tab
        press(tab, 'escape')
        assert not tab.claims_tab

    def test_description_is_prefilled_and_can_be_changed(self, tab, vault):
        press(tab, 'e')
        screen = draw(tab)
        assert "Marcus's mobile" in field(rows(screen), "Description") and waiting(screen) == ["Description"]
        press(tab, 'home', '\x0b')                   # clear the prefilled text
        type_text(tab, "Marcus mobile")
        press(tab, ENTER, 'n')                       # the new wording slugs differently: keep the id
        assert waiting(draw(tab)) == ["New value"]
        press(tab, ENTER)                            # empty value keeps the old one
        updated = vault.get("marcus-s-mobile")
        assert updated.description == "Marcus mobile" and updated.value == PHONE
        assert "marcus-s-mobile" in row_with(lines(tab), "Updated")

    def test_value_can_be_changed_and_the_new_one_is_masked(self, tab, vault):
        press(tab, 'down', 'e', ENTER)               # keep the description of the-model-y
        type_text(tab, "ABC 12D")
        press(tab, ENTER)
        assert vault.get("the-model-y").value == "ABC 12D"
        out = lines(tab)
        assert list_row(out, "the-model-y").endswith("ABC •••")
        assert "ABC 12D" not in "\n".join(out)

    def test_editing_a_revealed_secret_hides_the_new_value(self, tab, vault):
        # The one place a fresh value could stay painted on the list in the clear.
        press(tab, 'down', 'v')
        assert list_row(lines(tab), "the-model-y").endswith(PLATE)
        press(tab, 'e', ENTER)
        type_text(tab, "ABC 12D")
        press(tab, ENTER)
        out = lines(tab)
        assert vault.get("the-model-y").value == "ABC 12D"
        assert "ABC 12D" not in "\n".join(out)
        assert list_row(out, "the-model-y").endswith("ABC •••")

    def test_an_invalid_new_value_keeps_the_step_open(self, tab, vault):
        press(tab, 'down', 'e', ENTER)
        type_text(tab, "12")
        press(tab, ENTER)
        assert tab.captures_text is True
        screen = draw(tab)
        assert any("Cannot update" in l for l in rows(screen)) and waiting(screen) == ["New value"]
        assert vault.get("the-model-y").value == PLATE

    def test_enter_twice_without_changes_changes_nothing(self, tab, vault):
        before = vault.version
        press(tab, 'e', 'home', '\x0b', ENTER, ENTER)
        assert vault.version == before
        assert "Nothing changed" in "\n".join(lines(tab))

    def test_a_changed_description_offers_the_new_id_and_y_renames(self, tab, vault):
        view_hint = lines(tab)[-2]
        press(tab, 'e', 'home', '\x0b')
        type_text(tab, "Marcus mobile")
        press(tab, ENTER)
        screen = draw(tab)
        out = rows(screen)
        assert waiting(screen) == ["Id"]
        assert "marcus-mobile" in field(out, "Id")
        assert out[-2] != view_hint and "y" in out[-2].split()        # the hint row names the answer keys
        assert tab.captures_text is False
        press(tab, 'y')
        assert waiting(draw(tab)) == ["New value"]
        press(tab, ENTER)
        assert vault.get("marcus-s-mobile") is None
        renamed = vault.get("marcus-mobile")
        assert renamed.description == "Marcus mobile" and renamed.value == PHONE
        screen = draw(tab)
        status = row_with(rows(screen), "Renamed")
        assert "marcus-s-mobile" in status and "marcus-mobile" in status
        assert selected_id(screen) == "marcus-mobile"

    def test_n_keeps_the_id_but_still_saves_the_description(self, tab, vault):
        press(tab, 'e', 'home', '\x0b')
        type_text(tab, "Marcus mobile")
        press(tab, ENTER, 'n', ENTER)
        assert vault.get("marcus-s-mobile").description == "Marcus mobile"
        assert vault.get("marcus-mobile") is None
        assert "marcus-s-mobile" in row_with(lines(tab), "Updated")

    def test_a_declined_rename_alone_still_counts_as_a_change(self, tab, vault):
        # Only the wording changed and the id was kept: that is still a write
        press(tab, 'e', 'end')
        type_text(tab, "!")
        press(tab, ENTER)
        assert waiting(draw(tab)) == ["New value"]            # slug unchanged: no question asked
        press(tab, ENTER)
        assert vault.get("marcus-s-mobile").description == "Marcus's mobile!"

    def test_an_unchanged_description_asks_nothing(self, tab, vault):
        press(tab, 'e', ENTER)
        assert waiting(draw(tab)) == ["New value"]

    def test_a_taken_slug_is_offered_with_a_suffix(self, tab, vault):
        press(tab, 'e', 'home', '\x0b')
        type_text(tab, "Work e-mail")                # the-e-mail's id is taken by another secret
        press(tab, ENTER)
        screen = draw(tab)
        assert waiting(screen) == ["Id"] and "work-e-mail-2" in field(rows(screen), "Id")
        press(tab, 'y', ENTER)
        assert vault.get("work-e-mail-2").value == PHONE and vault.get("work-e-mail").value == EMAIL

    def test_escape_or_other_keys_at_the_rename_question(self, tab, vault):
        press(tab, 'e', 'home', '\x0b')
        type_text(tab, "Marcus mobile")
        press(tab, ENTER, 'x', 'q')
        screen = draw(tab)
        assert waiting(screen) == ["Id"] and "marcus-mobile" in field(rows(screen), "Id")
        before = vault.version
        press(tab, 'escape')
        assert vault.version == before and tab.captures_text is False
        assert waiting(draw(tab)) == []
        assert vault.get("marcus-s-mobile").description == "Marcus's mobile"


@pytest.mark.unit
class TestDeleteFlow:
    def test_d_then_y_deletes_the_selected_secret(self, tab, vault):
        press(tab, 'down', 'd')
        screen = draw(tab)
        assert len(waiting(screen)) == 1
        assert any("Delete" in l and "the-model-y" in l for l in rows(screen))
        press(tab, 'y')
        assert len(vault) == 2 and vault.get("the-model-y") is None
        screen = draw(tab)
        out = rows(screen)
        assert list_ids(out) == ["marcus-s-mobile", "work-e-mail"]
        assert "the-model-y" in row_with(out, "Deleted")
        assert selected_id(screen) == "work-e-mail"

    def test_d_then_escape_or_n_keeps_it(self, tab, vault):
        press(tab, 'd', 'escape')
        assert len(vault) == 3
        press(tab, 'd', 'n')
        assert len(vault) == 3
        assert not any("Delete" in l for l in lines(tab))

    def test_other_keys_do_not_confirm(self, tab, vault):
        press(tab, 'd', 'x', ENTER, 'Y')
        assert len(vault) == 3
        screen = draw(tab)
        assert len(waiting(screen)) == 1
        assert any("Delete" in l and "marcus-s-mobile" in l for l in rows(screen))


@pytest.mark.unit
class TestEmptyVault:
    def test_commands_needing_a_secret_are_consumed_without_a_crash_and_add_still_works(self, tmp_path):
        # A first-time user has nothing to edit; 'd' without the guard would enter the
        # confirm step with no target and the next frame would die drawing its form.
        tab = VaultTab(Vault(tmp_path / "vault.bin", "test"))
        for key in ('e', 'd', 'v'):
            assert tab.handle_input(key)
            assert tab.captures_text is False
            screen = draw(tab)
            assert waiting(screen) == []
            assert not any("Delete" in l for l in rows(screen))
        press(tab, 'a')
        assert "phone" in field(lines(tab), "Kind")


@pytest.mark.unit
class TestLocked:
    @pytest.fixture
    def locked(self, tmp_path):
        return VaultTab(Vault(tmp_path / "vault.bin", None))

    def test_the_table_tile_gives_the_reason_in_red_even_without_the_summary(self, locked):
        screen = draw(locked, height=24)
        out = rows(screen)
        reason = row_with(out, "vault.key not set")
        assert style_at(screen, out.index(reason), CONTENT).fg == Theme.COLOR_RED
        assert "locked" in "\n".join(out).lower()

    def test_edit_keys_are_ignored_but_consumed(self, locked):
        for key in ('a', 'e', 'd', 'v'):
            assert locked.handle_input(key)
            assert locked.captures_text is False
        screen = draw(locked)
        assert waiting(screen) == []
        assert not any("Delete" in l for l in rows(screen))

    def test_wrong_key_reports_the_decrypt_failure(self, tmp_path, vault):
        tab = VaultTab(Vault(vault.path, "wrong"))
        out = lines(tab)
        assert any("wrong vault.key" in l for l in out)
        assert list_rows(out) == []


@pytest.mark.unit
class TestConstruction:
    def test_constructing_the_tab_does_not_resolve_the_vault(self, monkeypatch):
        monkeypatch.setattr(Vault, "get_instance", classmethod(lambda cls: (_ for _ in ()).throw(AssertionError("touched"))))
        tab = VaultTab()
        assert tab.name == "Vault"
        assert tab.captures_text is False
