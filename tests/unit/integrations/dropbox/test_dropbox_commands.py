"""The Dropbox commands: the sign-in pair as the Volvo and Google ones work, and the search,
listing, read and find replies — a page under the history's cap with the call for the next
one in its header, the find pointing at a page, and the refusals as command failures."""
from unittest.mock import Mock, patch

import pytest

from assistant.commands.Commands import Commands
from assistant.integration.OAuthPaste import SignIn
from assistant.prompt.MessageHistory import MessageHistory
from integrations.dropbox.DocumentText import FIND_LIMIT
from integrations.dropbox.DropboxAuth import REDIRECT_URI
from integrations.dropbox.DropboxClient import DropboxError, Entry, Match
from integrations.dropbox.DropboxIntegration import Document, DropboxIntegration

MODULE = "integrations.dropbox.DropboxCommands"
URL = "https://www.dropbox.com/oauth2/authorize?client_id=app-key&state=st-1"
PASTED = f"{REDIRECT_URI}?code=c0de&state=st-1"
LEASE = Entry("/Household/Lease 2024.pdf", "Lease 2024.pdf", False, 1_258_291, 1709373600.0, "a" * 64)


def _integration(finish="", **attributes):
    integration = Mock(spec=DropboxIntegration)
    integration.login_begin.return_value = SignIn(URL, "st-1", REDIRECT_URI)
    integration.login_finish.return_value = finish
    for name, value in attributes.items():
        setattr(integration, name, value)
    return integration


def _monitor(integration):
    monitor = Mock()
    monitor.find.return_value = integration
    return monitor


def _with(integration):
    return patch.object(Commands, "_integration_monitor", return_value=_monitor(integration))


def owner(commands_instance, command: str):
    return commands_instance._owner(command)


def long_document(paragraphs: int = 40) -> Document:
    text = "\n\n".join(f"[p. {i}]\nClause {i}: " + " ".join(f"word{i}-{j}" for j in range(70)) for i in range(1, paragraphs + 1))
    return Document(LEASE, text, "PDF", False)


@pytest.mark.unit
class TestLogin:
    def test_dropbox_login_runs_on_the_config_lane_and_relays_the_link(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("dropbox_login", [""]) == ""
        bg.assert_called_once_with(owner(commands_instance, "dropbox_login")._run_dropbox_login, (False,), lane=Commands.CONFIG_LANE)

        integration = _integration()
        with _with(integration), patch(f"{MODULE}.system_note") as note:
            reply = owner(commands_instance, "dropbox_login")._run_dropbox_login()
        assert URL in reply and "dropbox_login_code" in reply and "localhost:18724" in reply
        assert note.call_args.args[1] == reply

        with _with(integration), patch(f"{MODULE}.system_note") as note:
            owner(commands_instance, "dropbox_login")._run_dropbox_login(from_chat=True)
        note.assert_not_called()

    def test_a_disabled_or_unconfigured_integration_says_what_to_do(self, commands_instance):
        disabled = Mock(spec=["state"])
        disabled.state = Mock(value="disabled")
        with _with(disabled):
            assert 'enable_integration("Dropbox")' in owner(commands_instance, "dropbox_login")._run_dropbox_login()
        not_configured = Mock(spec=["state"])
        not_configured.state = Mock(value="not_configured")
        with _with(not_configured):
            assert "dropbox.app_key" in owner(commands_instance, "search_dropbox")._run_search("x", "")

    def test_dropbox_login_code_takes_the_whole_address_and_rechecks_the_integration(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("dropbox_login_code", [f'"{PASTED}"']) == ""
        bg.assert_called_once_with(owner(commands_instance, "dropbox_login_code")._run_dropbox_login_code, (PASTED, False), lane=Commands.CONFIG_LANE)
        assert "paste" in commands_instance.run("dropbox_login_code", [""])

        integration = _integration()
        monitor = _monitor(integration)
        with patch.object(Commands, "_integration_monitor", return_value=monitor), patch(f"{MODULE}.system_note") as note:
            reply = owner(commands_instance, "dropbox_login_code")._run_dropbox_login_code(PASTED)
        assert "signed in" in reply and note.call_args.args[1] == reply
        integration.login_finish.assert_called_once_with(PASTED)
        monitor.check_now.assert_called_once_with("Dropbox")

    def test_a_failed_finish_is_told(self, commands_instance):
        with _with(_integration(finish="the code expired")), patch(f"{MODULE}.system_note"):
            reply = owner(commands_instance, "dropbox_login_code")._run_dropbox_login_code(PASTED)
        assert "did not complete" in reply and "expired" in reply


@pytest.mark.unit
class TestSearch:
    def test_the_arguments_reach_the_background_unquoted(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("search_dropbox", ['"lease deposit"', '"/Household"']) == ""
        bg.assert_called_once_with(owner(commands_instance, "search_dropbox")._run_search, ("lease deposit", "/Household"))
        assert "words" in commands_instance.run("search_dropbox", [""])

    def test_matches_are_listed_with_size_date_and_highlight(self, commands_instance):
        integration = _integration()
        integration.search.return_value = ([Match(LEASE, "content", "deposit"),
                                            Match(Entry("/Household/Invoices", "Invoices", True), "filename", "")], True)
        with _with(integration):
            reply = owner(commands_instance, "search_dropbox")._run_search("deposit", "/Household")
        lines = reply.splitlines()
        assert lines[0].startswith('2 match(es) for "deposit" in /Household')
        assert "/Household/Lease 2024.pdf" in lines[1] and "1.2 MB" in lines[1] and "2024-03-02" in lines[1] and "deposit" in lines[1]
        assert lines[2] == "/Household/Invoices/  (folder)"
        assert "narrow" in lines[3]
        integration.search.assert_called_once_with("deposit", "/Household")

    def test_nothing_found_points_at_the_other_ways_in(self, commands_instance):
        integration = _integration()
        integration.search.return_value = ([], False)
        with _with(integration):
            reply = owner(commands_instance, "search_dropbox")._run_search("unicorn", "")
        assert "No file" in reply and "find_in_dropbox_file" in reply and "list_dropbox_folder" in reply

    def test_a_refused_folder_is_a_failure(self, commands_instance):
        integration = _integration()
        integration.search.side_effect = ValueError("/Work is outside the folders the assistant may read: /Household")
        with _with(integration):
            assert owner(commands_instance, "search_dropbox")._run_search("x", "/Work").startswith("search_dropbox failed: /Work is outside")


@pytest.mark.unit
class TestList:
    def test_entries_are_listed_folders_as_such(self, commands_instance):
        integration = _integration()
        integration.list_folder.return_value = ([Entry("/Household/Invoices", "Invoices", True), LEASE], False)
        with _with(integration):
            reply = owner(commands_instance, "list_dropbox_folder")._run_list('"/Household"')
        assert reply.splitlines() == ["/Household: 2 entries", "/Household/Invoices/", "/Household/Lease 2024.pdf  (1.2 MB, 2024-03-02)"]

    def test_an_empty_folder_and_a_missing_one(self, commands_instance):
        integration = _integration()
        integration.list_folder.return_value = ([], False)
        with _with(integration):
            assert owner(commands_instance, "list_dropbox_folder")._run_list("") == "/ is empty."
        integration.list_folder.side_effect = DropboxError(409, "path/not_found/..", "nothing at that path")
        with _with(integration):
            assert "nothing at that path" in owner(commands_instance, "list_dropbox_folder")._run_list("/Gone")


@pytest.mark.unit
class TestRead:
    def test_the_path_and_the_page_reach_the_background_and_a_bad_page_is_refused_at_once(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("read_dropbox_file", ['"/Household/Lease 2024.pdf"', "3"]) == ""
            assert commands_instance.run("read_dropbox_file", ['"/a.txt"']) == ""
        assert [c.args[1] for c in bg.call_args_list] == [("/Household/Lease 2024.pdf", 3), ("/a.txt", 1)]
        assert "number" in commands_instance.run("read_dropbox_file", ["/a.txt", "three"])
        assert "path" in commands_instance.run("read_dropbox_file", [""])

    def test_a_page_stays_under_the_history_cap_and_its_header_names_the_next_one(self, commands_instance):
        integration = _integration()
        integration.document.return_value = long_document()
        with _with(integration):
            reply = owner(commands_instance, "read_dropbox_file")._run_read(LEASE.path, 2)
        header, body = reply.split("\n---\n", 1)
        assert len(reply) < MessageHistory.COMMAND_RESULT_CHARS, "the history stores the page whole"
        assert header.startswith("«Lease 2024.pdf» — page 2 of ")
        assert f'read_dropbox_file("{LEASE.path}", 3)' in header and "find_in_dropbox_file" in header
        assert "1.2 MB PDF" in header and "2024-03-02" in header
        assert body.startswith("[p. ") and "Clause 1:" not in body

    def test_the_last_page_has_no_next_and_a_short_file_is_one_page(self, commands_instance):
        integration = _integration()
        integration.document.return_value = long_document()
        with _with(integration):
            first = owner(commands_instance, "read_dropbox_file")._run_read(LEASE.path, 1)
            pages = int(first.split(" of ", 1)[1].split(" ", 1)[0])
            last = owner(commands_instance, "read_dropbox_file")._run_read(LEASE.path, pages)
        assert "Next:" not in last and f"page {pages} of {pages}" in last

        integration.document.return_value = Document(LEASE, "Short lease.", "PDF", True)
        with _with(integration):
            short = owner(commands_instance, "read_dropbox_file")._run_read(LEASE.path, 1)
        assert "page 1 of 1" in short and "Next:" not in short and "find_in_dropbox_file" not in short

    def test_a_document_without_text_says_so_instead_of_an_empty_page(self, commands_instance):
        integration = _integration()
        integration.document.return_value = Document(LEASE, "   \n", "PDF", False)
        with _with(integration):
            reply = owner(commands_instance, "read_dropbox_file")._run_read(LEASE.path, 1)
        assert "no text" in reply and "---" not in reply and "Lease 2024.pdf" in reply

    def test_page_zero_and_a_negative_page_are_refused_like_one_past_the_end(self, commands_instance):
        integration = _integration()
        integration.document.return_value = Document(LEASE, "Short lease.", "PDF", True)
        with _with(integration):
            for number in (0, -2):
                reply = owner(commands_instance, "read_dropbox_file")._run_read(LEASE.path, number)
                assert reply.startswith("read_dropbox_file failed") and f"page {number}" in reply and "---" not in reply

    def test_a_page_past_the_end_says_how_many_there_are(self, commands_instance):
        integration = _integration()
        integration.document.return_value = Document(LEASE, "Short lease.", "PDF", True)
        with _with(integration):
            reply = owner(commands_instance, "read_dropbox_file")._run_read(LEASE.path, 7)
        assert reply.startswith("read_dropbox_file failed") and "1 page(s)" in reply and "page 7" in reply

    def test_an_unreadable_file_is_a_failure_with_the_reason(self, commands_instance):
        integration = _integration()
        integration.document.side_effect = ValueError("photo.jpg (4 KB) cannot be read: a .jpg file has no text to read")
        with _with(integration):
            reply = owner(commands_instance, "read_dropbox_file")._run_read("/photo.jpg", 1)
        assert reply.startswith("read_dropbox_file failed") and ".jpg" in reply


@pytest.mark.unit
class TestFind:
    def test_the_words_after_the_path_are_joined(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("find_in_dropbox_file", ['"/a.pdf"', '"deposit"', "returned"]) == ""
        assert bg.call_args.args[1] == ("/a.pdf", "deposit returned")
        assert "words" in commands_instance.run("find_in_dropbox_file", ["/a.pdf"])

    def test_hits_name_the_page_to_read_and_the_pdf_page(self, commands_instance):
        integration = _integration()
        integration.document.return_value = long_document()
        with _with(integration):
            reply = owner(commands_instance, "find_in_dropbox_file")._run_find(LEASE.path, "clause 33:")
        lines = reply.splitlines()
        assert lines[0].startswith('«Lease 2024.pdf» — 1 line(s) match "clause 33:"') and "read_dropbox_file" in lines[0]
        assert lines[1].startswith("page ") and "[p. 33]" in lines[1] and "Clause 33:" in lines[1]
        page = int(lines[1].split()[1])
        assert page > 1, "the hit is deep in the document, not on the first page"

    def test_no_hit_says_so_and_the_cap_is_told(self, commands_instance):
        integration = _integration()
        integration.document.return_value = long_document()
        with _with(integration):
            assert "no line" in owner(commands_instance, "find_in_dropbox_file")._run_find(LEASE.path, "unicorn")
            many = owner(commands_instance, "find_in_dropbox_file")._run_find(LEASE.path, "clause")
        assert f"the first {FIND_LIMIT}" in many.splitlines()[0] and len(many.splitlines()) == FIND_LIMIT + 1
