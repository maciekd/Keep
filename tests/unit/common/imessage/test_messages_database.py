"""MessagesDatabase over a chat.db of Apple's shape on disk: the rows after a ROWID, what a row
carries, the Send Later columns, attachments, group chats and what the database knows of a handle."""
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from common.imessage import MessagesDatabase as module
from common.imessage.MessagesDatabase import MessagesDatabase
from common.imessage.MessagesHost import GroupChat, NO_ACCESS, NO_DATABASE, SCHEDULE_COMPLETED, SCHEDULE_FAILED

from .messages_fakes import ChatDb, GROUP, NOW, apple

DAY = 86400
NAMES = {"+46701740605": "Alice", "+46701740606": "Bob"}


def name(handle):
    return NAMES.get(handle, handle or "Unknown")


@pytest.fixture
def chat(tmp_path):
    """A chat.db with Alice (handle 1) in a 1:1 chat (1), Alice and Bob in the group "Family" (2)
    and in an unnamed group (3)."""
    db = ChatDb(tmp_path / "chat.db")
    db.handle(1, "+46701740605").handle(2, "+46701740606")
    db.chat(1, "any;-;+46701740605", [1])
    db.chat(2, "any;+;chat2", [1, 2], name="Family", style=GROUP)
    db.chat(3, "any;+;chat3", [1, 2], style=GROUP)
    return db


def read(db, after=0, horizon=None, limit=500):
    return MessagesDatabase(db.path).messages_after(after, horizon, limit, name)


@pytest.mark.unit
class TestMessagesAfter:
    def test_an_incoming_row_carries_its_sender_by_handle_and_by_name(self, chat):
        chat.message(7, "  Hello \n", ts=NOW - 60, chat=1, sender=1)
        (message,), cursor, scanned = read(chat)
        assert (message.rowid, message.text, message.is_from_me) == (7, "Hello", False)
        assert (message.handle, message.sender) == ("+46701740605", "Alice")
        assert message.timestamp == pytest.approx(NOW - 60, abs=1e-3)
        assert message.chat_name is None and message.recipients == []
        assert (cursor, scanned) == (7, 1)

    def test_a_sender_no_contact_has_is_named_by_the_handle(self, chat):
        chat.handle(9, "+46701740699").message(1, "Hi", chat=1, sender=9)
        assert read(chat)[0][0].sender == "+46701740699"

    def test_a_group_row_names_the_group(self, chat):
        chat.message(1, "Hey all", chat=2, sender=1)
        assert read(chat)[0][0].chat_name == "Family"

    def test_an_outgoing_row_names_the_chats_participants_since_the_row_has_no_handle(self, chat):
        chat.message(1, "On my way", chat=1).message(2, "On my way, both", chat=3)
        direct, group = read(chat)[0]
        assert direct.is_from_me and direct.handle is None
        assert direct.recipients == ["Alice"]
        assert group.recipients == ["Alice", "Bob"]

    def test_an_outgoing_row_of_a_named_group_needs_no_participants(self, chat):
        chat.message(1, "Dinner at 7", chat=2)
        (message,), _, _ = read(chat)
        assert message.chat_name == "Family" and message.recipients == []

    def test_an_outgoing_row_without_a_chat_has_no_recipients(self, chat):
        chat.message(1, "Into the void")
        assert read(chat)[0][0].recipients == []

    def test_participants_without_a_handle_id_are_left_out(self, chat):
        chat.handle(8, None).chat(4, "any;-;none", [8]).message(1, "Hello?", chat=4)
        assert read(chat)[0][0].recipients == []

    def test_a_row_with_neither_text_nor_a_file_is_left_out_and_the_cursor_passes_it(self, chat):
        chat.message(1, "\n", chat=1, sender=1).message(2, None, chat=1, sender=1).message(3, "Hi", chat=1, sender=1)
        messages, cursor, scanned = read(chat)
        assert [m.rowid for m in messages] == [3]
        assert (cursor, scanned) == (3, 3)

    def test_trailing_empty_rows_still_move_the_cursor(self, chat):
        chat.message(1, "Hi", chat=1, sender=1).message(2, "", chat=1, sender=1)
        assert read(chat)[1] == 2

    def test_system_messages_are_not_read(self, chat):
        chat.message(1, "Alice named the conversation", chat=2, sender=1, system=True)
        assert read(chat) == ([], 0, 0)

    def test_the_text_comes_from_the_attributed_body_when_the_column_is_empty(self, chat):
        chat.message(1, None, chat=1, sender=1, attributedBody=b"typedstream")
        with patch.object(module, "attributed_text", return_value=" From the blob ") as decode:
            (message,), _, _ = read(chat)
        assert message.text == "From the blob"
        decode.assert_called_once_with(b"typedstream")

    def test_only_rows_after_the_cursor_up_to_the_limit_are_read(self, chat):
        for rowid in range(1, 7):
            chat.message(rowid, f"m{rowid}", chat=1, sender=1)
        messages, cursor, scanned = read(chat, after=2, limit=3)
        assert [m.rowid for m in messages] == [3, 4, 5]
        assert (cursor, scanned) == (5, 3)

    def test_nothing_new_leaves_the_cursor_where_it_was(self, chat):
        chat.message(1, "Hi", chat=1, sender=1)
        assert read(chat, after=4) == ([], 4, 0)


@pytest.mark.unit
class TestHorizon:
    """ROWIDs are not in date order — Messages in iCloud writes the history in chunks — so the
    horizon is a test on the date of every row, never on a ROWID."""

    HORIZON = NOW - 90 * DAY

    @pytest.fixture
    def shuffled(self, chat):
        for rowid, ts in [(1, NOW - 10 * DAY), (2, self.HORIZON - 400 * DAY), (3, self.HORIZON + 5 * DAY),
                          (4, self.HORIZON - DAY), (5, NOW - 2 * DAY), (6, self.HORIZON - 2000 * DAY)]:
            chat.message(rowid, f"m{rowid}", ts=ts, chat=1, sender=1)
        return MessagesDatabase(chat.path)

    def test_rows_dated_before_the_horizon_are_skipped_wherever_they_sit(self, shuffled):
        messages, cursor, scanned = shuffled.messages_after(0, self.HORIZON, 500)
        assert [m.rowid for m in messages] == [1, 3, 5]
        assert (cursor, scanned) == (5, 3), "the old row written last is never read"

    def test_the_start_is_just_before_the_first_row_inside_the_horizon(self, shuffled):
        assert shuffled.cursor_at(self.HORIZON) == 0
        assert shuffled.cursor_at(NOW - 5 * DAY) == 4

    def test_a_horizon_past_every_row_starts_at_the_end_of_the_table(self, shuffled):
        assert shuffled.cursor_at(NOW + DAY) == 6

    def test_deleting_the_newest_message_does_not_move_the_end_of_the_table_back(self, shuffled, tmp_path):
        """The end is the highest ROWID ever given out, as chat.db's own counter keeps it: a
        cursor that stood on a message deleted since is still a place in this database."""
        import sqlite3
        conn = sqlite3.connect(shuffled.path)
        conn.execute("DELETE FROM message WHERE ROWID >= 5")
        conn.commit()
        conn.close()
        assert shuffled.cursor_at(NOW + DAY) == 6

    def test_the_end_of_the_table_is_its_counter_whatever_a_row_is_dated(self, shuffled):
        """Asked by a date instead — "past every message" — one row dated after that date would
        end the table just before itself."""
        import sqlite3
        conn = sqlite3.connect(shuffled.path)
        conn.execute("UPDATE message SET date = ? WHERE ROWID = 2", (apple(NOW + 200 * 365 * DAY),))
        conn.commit()
        conn.close()
        assert shuffled.last_rowid() == 6

    def test_a_table_without_the_counter_ends_at_its_last_row(self, tmp_path):
        import sqlite3
        path = tmp_path / "old.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE message (ROWID INTEGER PRIMARY KEY, date INTEGER)")
        conn.executemany("INSERT INTO message (ROWID, date) VALUES (?, 0)", [(3,), (9,)])
        conn.commit()
        conn.close()
        assert MessagesDatabase(str(path)).cursor_at(NOW + DAY) == 9
        assert MessagesDatabase(str(path)).last_rowid() == 9

    def test_the_count_ahead_is_what_a_read_will_return(self, shuffled):
        assert shuffled.count_after(0, self.HORIZON) == 3
        assert shuffled.count_after(3, self.HORIZON) == 1
        assert shuffled.count_after(5, self.HORIZON) == 0

    def test_without_a_horizon_every_row_counts(self, shuffled):
        assert shuffled.count_after(0, None) == 6
        assert len(shuffled.messages_after(0, None, 500)[0]) == 6

    def test_an_empty_table_starts_at_zero(self, tmp_path):
        assert MessagesDatabase(ChatDb(tmp_path / "chat.db").path).cursor_at(NOW) == 0


@pytest.mark.unit
class TestSendLater:
    """chat.db gets a Send Later row at compose time, marked schedule_type=2 until Apple sends it."""

    def test_a_marked_row_comes_as_send_later_without_its_body(self, chat):
        chat.message(5, "See you tomorrow", ts=NOW + 3600, chat=1, schedule_type=2)
        (message,), cursor, _ = read(chat)
        assert message.send_later and message.is_from_me and message.in_chat
        assert message.text is None and message.schedule_state == ""
        assert message.timestamp == pytest.approx(NOW + 3600, abs=1e-3), "the time it is set to send"
        assert cursor == 5

    def test_a_chat_less_incoming_copy_is_told_apart(self, chat):
        chat.message(9, "Hej", sender=1, schedule_type=2, schedule_state=3)
        (message,), _, _ = read(chat)
        assert message.send_later and not message.is_from_me and not message.in_chat
        assert message.schedule_state == SCHEDULE_COMPLETED

    def test_every_marked_row_is_listed_with_its_state(self, chat):
        chat.message(1, "plain", chat=1, sender=1)
        chat.message(2, "later", ts=NOW + 60, chat=1, schedule_type=2)
        chat.message(3, "failed", chat=1, schedule_type=2, schedule_state=4)
        rows = {m.rowid: m for m in MessagesDatabase(chat.path).send_later()}
        assert set(rows) == {2, 3}
        assert rows[3].schedule_state == SCHEDULE_FAILED and rows[2].schedule_state == ""
        assert rows[2].in_chat and rows[2].is_from_me and rows[2].send_later

    def test_a_tracked_row_is_read_again_as_it_is_now(self, chat):
        """Apple resets schedule_type when the message sends: the same row, no longer marked."""
        chat.message(5, " See you tomorrow ", chat=1, schedule_type=2)
        database = MessagesDatabase(chat.path)
        (pending,) = database.messages([5], name)
        assert pending.send_later and pending.text == "See you tomorrow"
        chat.execute("UPDATE message SET schedule_type = 0 WHERE ROWID = 5")
        (sent,) = database.messages([5], name)
        assert not sent.send_later and sent.recipients == ["Alice"]

    def test_a_deleted_row_is_missing_from_the_answer(self, chat):
        chat.message(5, "kept", chat=1, schedule_type=2)
        assert [m.rowid for m in MessagesDatabase(chat.path).messages([5, 6], name)] == [5]

    def test_no_rowids_reads_nothing(self, tmp_path):
        assert MessagesDatabase(tmp_path / "missing.db").messages([], name) == []


@pytest.mark.unit
class TestScheduleColumns:
    """A chat.db from before macOS 15 has no Send Later columns; it is read as plain messages
    instead of failing on every poll (a beta tester's Sonoma Mac, 2026-10-16)."""

    @pytest.fixture
    def sonoma(self, tmp_path):
        db = ChatDb(tmp_path / "chat.db", send_later_columns=False)
        db.handle(1, "+46701740605").chat(1, "any;-;+46701740605", [1])
        db.message(1, "Hello from Alice", chat=1, sender=1)
        return db

    def test_a_sonoma_schema_has_no_send_later_columns(self, sonoma):
        assert MessagesDatabase(sonoma.path).probe_schema() is False

    def test_a_sequoia_schema_has_both(self, chat):
        assert MessagesDatabase(chat.path).probe_schema() is True

    def test_one_column_alone_is_not_enough(self, sonoma):
        sonoma.execute("ALTER TABLE message ADD COLUMN schedule_type INTEGER DEFAULT 0")
        assert MessagesDatabase(sonoma.path).probe_schema() is False

    def test_without_the_columns_the_rows_are_still_read(self, sonoma):
        database = MessagesDatabase(sonoma.path)
        database.probe_schema()
        (message,), cursor, _ = database.messages_after(0, None, 500, name)
        assert message.text == "Hello from Alice" and not message.send_later
        assert cursor == 1
        assert [m.text for m in database.messages([1], name)] == ["Hello from Alice"]

    def test_without_the_columns_no_scheduled_message_is_looked_for(self, sonoma):
        database = MessagesDatabase(sonoma.path)
        database.probe_schema()
        with patch.object(module.sqlite3, "connect") as connect:
            assert database.send_later() == []
        connect.assert_not_called()

    def test_a_schema_that_cannot_be_read_keeps_the_columns_in_the_query(self, chat):
        """Not reading the schema is not evidence of a Sonoma database: the query keeps naming
        the columns, so a Sequoia Mac with a momentary lock keeps its send-later tracking and
        the read, not the schema probe, reports what is really wrong."""
        database = MessagesDatabase(chat.path)
        with patch.object(module.sqlite3, "connect", side_effect=sqlite3.OperationalError("database is locked")):
            assert database.probe_schema() is True
        assert "m.schedule_type" in database._schedule_select()


@pytest.mark.unit
class TestAttachments:
    @pytest.fixture
    def photo(self, tmp_path):
        path = tmp_path / "IMG_0001.jpg"
        path.write_bytes(b"fake jpeg")
        return path

    def test_a_file_on_disk_crosses_as_a_reference_not_as_a_path(self, chat, photo):
        chat.message(1, None, chat=1, sender=1).attachment(40, 1, str(photo), "image/jpeg")
        (message,), _, _ = read(chat)
        (attachment,) = message.attachments
        assert (attachment.id, attachment.name, attachment.mime_type) == (40, "IMG_0001.jpg", "image/jpeg")
        assert str(photo.parent) not in repr(message)

    def test_the_reference_gives_the_bytes_back(self, chat, photo):
        chat.message(1, None, chat=1, sender=1).attachment(40, 1, str(photo), "image/jpeg")
        assert MessagesDatabase(chat.path).read_attachment(40) == b"fake jpeg"

    def test_an_id_chat_db_does_not_list_reads_nothing(self, chat):
        with pytest.raises(FileNotFoundError):
            MessagesDatabase(chat.path).read_attachment(41)
        with pytest.raises(FileNotFoundError):
            MessagesDatabase(chat.path).attachment_size(41)

    def test_the_size_is_told_without_opening_the_file(self, chat, photo):
        chat.message(1, None, chat=1, sender=1).attachment(40, 1, str(photo), "image/jpeg")
        with patch.object(module, "open", side_effect=AssertionError("the file was opened"), create=True):
            assert MessagesDatabase(chat.path).attachment_size(40) == len(b"fake jpeg")

    def test_every_type_of_file_is_listed(self, chat, tmp_path):
        """What counts as media is the reader's to say; the database lists the files."""
        voice = tmp_path / "voice.m4a"
        voice.write_bytes(b"audio")
        chat.message(1, "listen", chat=1, sender=1).attachment(40, 1, str(voice), "audio/mp4")
        assert [a.mime_type for a in read(chat)[0][0].attachments] == ["audio/mp4"]

    def test_a_missing_file_is_skipped(self, chat):
        chat.message(1, "look", chat=1, sender=1).attachment(40, 1, "/nonexistent/photo.jpg", "image/jpeg")
        assert read(chat)[0][0].attachments == []

    def test_a_file_that_is_not_on_disk_yet_is_reported_and_its_row_is_not_passed_over(self, chat, tmp_path):
        """Messages writes the row before a large attachment has finished downloading: a
        reader that wants the picture must be able to tell, and to find the row again."""
        late = tmp_path / "IMG_0002.heic"
        chat.message(1, None, chat=1, sender=1).attachment(40, 1, str(late), "image/heic")
        (message,), cursor, _ = read(chat)
        assert (message.rowid, message.attachments, message.pending_attachments, cursor) == (1, [], 1, 1)
        late.write_bytes(b"fake heic")
        (message,), _, _ = read(chat)
        assert ([a.id for a in message.attachments], message.pending_attachments) == ([40], 0)

    def test_the_mark_that_stands_for_an_attachment_is_no_text(self, chat, photo):
        """chat.db's body holds U+FFFC where a file is attached; a row with nothing else and
        no file has nothing to say."""
        chat.message(1, "\ufffc", chat=1, sender=1).attachment(40, 1, str(photo), "image/jpeg")
        chat.message(2, "\ufffcwhose bike is this?", chat=1, sender=1).attachment(41, 2, str(photo), "image/jpeg")
        chat.message(3, " \ufffc ", chat=1, sender=1)
        rows = {m.rowid: m.text for m in read(chat)[0]}
        assert rows == {1: "", 2: "whose bike is this?"}

    def test_an_app_payload_that_is_not_on_disk_is_not_a_file_to_wait_for(self, chat):
        chat.message(1, "https://keep.example", chat=1, sender=1).attachment(40, 1, "/nonexistent/x.pluginPayloadAttachment", None)
        assert read(chat)[0][0].pending_attachments == 0

    def test_an_app_payload_is_never_a_file(self, chat, tmp_path):
        payload = tmp_path / "preview.pluginPayloadAttachment"
        payload.write_bytes(b"link preview")
        chat.message(1, "https://keep.example", chat=1, sender=1).attachment(40, 1, str(payload), None)
        assert read(chat)[0][0].attachments == []

    def test_a_tilde_path_is_read_from_the_home_folder(self, chat, tmp_path, photo):
        chat.message(1, None, chat=1, sender=1).attachment(40, 1, "~/IMG_0001.jpg", "image/jpeg")
        with patch.object(Path, "home", return_value=tmp_path):
            (message,), _, _ = read(chat)
            assert MessagesDatabase(chat.path).read_attachment(40) == b"fake jpeg"
        assert [a.name for a in message.attachments] == ["IMG_0001.jpg"]

    def test_a_row_with_only_a_file_is_returned(self, chat, photo):
        chat.message(1, None, chat=1, sender=1).attachment(40, 1, str(photo), "image/jpeg")
        assert [m.rowid for m in read(chat)[0]] == [1]


@pytest.mark.unit
class TestCheck:
    def test_a_readable_database_is_fine(self, chat):
        assert MessagesDatabase(chat.path).check() == ("", "")

    def test_a_missing_file_means_messages_was_never_set_up(self, tmp_path):
        reason, detail = MessagesDatabase(tmp_path / "chat.db").check()
        assert reason == NO_DATABASE and str(tmp_path / "chat.db") in detail

    def test_a_file_sqlite_is_refused_is_an_access_problem(self, chat):
        """sqlite says "unable to open database file" whether the file is missing or macOS
        refused it; the file is there, so it is the refusal."""
        with patch.object(module.sqlite3, "connect",
                          side_effect=sqlite3.OperationalError("unable to open database file")):
            reason, detail = MessagesDatabase(chat.path).check()
        assert reason == NO_ACCESS and chat.path in detail and "unable to open" in detail

    def test_any_other_failure_is_reported_not_raised(self, tmp_path):
        garbage = tmp_path / "chat.db"
        garbage.write_bytes(b"this is not a database " * 64)
        assert MessagesDatabase(garbage).check()[0] == NO_ACCESS


def _activity_db(path, handles, chats):
    """Write a chat.db with Apple's shape: `handles` are (rowid, id, service); `chats` are
    (chat_id, [handle rowids], [(message rowid, date, is_from_me)]). Outgoing messages carry
    handle_id = 0 as Apple writes them."""
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT, service TEXT);
            CREATE TABLE message (ROWID INTEGER PRIMARY KEY, date INTEGER, is_from_me INTEGER, handle_id INTEGER);
            CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER);
            CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
        """)
        conn.executemany("INSERT INTO handle VALUES (?, ?, ?)", handles)
        for chat_id, members, messages in chats:
            conn.executemany("INSERT INTO chat_handle_join VALUES (?, ?)", [(chat_id, h) for h in members])
            for rowid, date, from_me in messages:
                sender = 0 if from_me else members[0]
                conn.execute("INSERT INTO message VALUES (?, ?, ?, ?)", (rowid, date, from_me, sender))
                conn.execute("INSERT INTO chat_message_join VALUES (?, ?)", (chat_id, rowid))


@pytest.mark.unit
class TestChatActivity:
    def _activity(self, tmp_path, handles, chats, wanted):
        db = tmp_path / "chat.db"
        _activity_db(db, handles, chats)
        return MessagesDatabase(db).chat_activity(wanted)

    def test_conversation_where_only_the_owner_writes_counts(self, tmp_path):
        """The owner texts the phone daily and the contact never replies; the email got one reply
        years ago. Counting by message.handle_id (0 on outgoing rows) would hand the email the win."""
        activity = self._activity(
            tmp_path,
            handles=[(1, "+46701740605", "iMessage"), (2, "carol@example.com", "iMessage")],
            chats=[(10, [1], [(100, 900, 1), (101, 901, 1), (102, 902, 1)]),
                   (11, [2], [(103, 500, 0)])],
            wanted={"+46701740605", "carol@example.com"})
        assert activity["+46701740605"] == (True, 902, 3)
        assert activity["carol@example.com"] == (True, 500, 1)

    def test_imessage_and_sms_rows_of_one_handle_merge(self, tmp_path):
        activity = self._activity(
            tmp_path,
            handles=[(1, "+46701740605", "SMS"), (2, "+46701740605", "iMessage")],
            chats=[(10, [1], [(100, 900, 0), (101, 300, 1)]), (11, [2], [(102, 500, 1)])],
            wanted={"+46701740605"})
        assert activity["+46701740605"] == (True, 900, 3)

    def test_phone_indexed_by_suffix_and_email_lowercased(self, tmp_path):
        activity = self._activity(
            tmp_path,
            handles=[(1, "+46701740605", "iMessage"), (2, "Carol@Example.com", "iMessage")],
            chats=[(10, [1], [(100, 1, 0)]), (11, [2], [(101, 2, 0), (102, 2, 1)])],
            wanted={"0701740605", "carol@example.com"})
        assert activity["701740605"] == (True, 1, 1)
        assert activity["carol@example.com"] == (True, 2, 2)
        assert "example.com" not in activity

    def test_only_the_wanted_handles_are_read(self, tmp_path):
        activity = self._activity(
            tmp_path,
            handles=[(1, "+46701740605", "iMessage"), (2, "+12025550176", "iMessage"), (3, "bob@example.com", "SMS")],
            chats=[(10, [1, 2], [(100, 1, 0)]), (11, [3], [(101, 5, 1)])],
            wanted={"+46701740605"})
        assert set(activity) == {"+46701740605", "701740605"}

    def test_handle_without_messages_still_known_to_imessage(self, tmp_path):
        activity = self._activity(tmp_path, handles=[(1, "+46701740605", "iMessage")], chats=[],
                                  wanted={"+46701740605"})
        assert activity["+46701740605"] == (True, 0, 0)

    def test_more_handles_than_one_query_holds(self, tmp_path):
        many = [(i, f"+4670174{i:04d}", "iMessage") for i in range(1, 1203)]
        chats = [(1000 + i, [i], [(100_000 + i, i, 1)]) for i, _, _ in many]
        activity = self._activity(tmp_path, many, chats, wanted={h for _, h, _ in many})
        assert activity["+46701741202"] == (True, 1202, 1)
        assert activity["+46701740001"] == (True, 1, 1)

    def test_an_unreadable_db_knows_nothing(self, tmp_path):
        assert MessagesDatabase(tmp_path / "missing.db").chat_activity({"+46701740605"}) == {}


@pytest.mark.unit
class TestWhatAChatConsumerReads:
    """The columns a consumer that answers in the chat needs: message.guid and message.service,
    chat.guid and chat.style, message.thread_originator_guid (the bubble a reply's thread starts
    at) and message.associated_message_type / associated_message_guid (a tapback and its bubble)."""

    def test_a_row_says_which_chat_it_is_in_and_whether_that_is_a_group(self, chat):
        chat.message(1, "Hi", chat=1, sender=1, guid="G-1", service="iMessage")
        chat.message(2, "Hi all", chat=2, sender=2, guid="G-2", service="SMS")
        direct, group = read(chat)[0]
        assert (direct.guid, direct.chat_guid, direct.is_group, direct.service) == \
            ("G-1", "any;-;+46701740605", False, "iMessage")
        assert (group.guid, group.chat_guid, group.is_group, group.service) == ("G-2", "any;+;chat2", True, "SMS")

    def test_a_reply_in_the_thread_of_the_accounts_own_bubble_names_it(self, chat):
        chat.message(1, "Dinner is at seven", chat=2, guid="OWN-1")
        chat.message(2, "and dessert?", chat=2, sender=1, guid="G-2", thread_originator_guid="OWN-1")
        reply = read(chat)[0][1]
        assert (reply.reply_to, reply.reply_to_from_me, reply.reply_to_text) == ("OWN-1", True, "Dinner is at seven")
        assert reply.tapback == 0

    def test_a_reply_to_somebody_elses_bubble_is_not_one_to_the_account(self, chat):
        chat.message(1, "Who is cooking?", chat=2, sender=2, guid="BOB-1")
        chat.message(2, "me", chat=2, sender=1, guid="G-2", thread_originator_guid="BOB-1")
        reply = read(chat)[0][1]
        assert (reply.reply_to, reply.reply_to_from_me, reply.reply_to_text) == ("BOB-1", False, "Who is cooking?")

    def test_a_reply_names_no_bubble_of_another_chat(self, chat):
        """The guid a reply carries is the sender's to write: one that is a bubble of another
        conversation must not bring that conversation's text into this one."""
        chat.message(1, "The alarm code is 4711", chat=1, guid="OWN-1")
        chat.message(2, "what was that?", chat=2, sender=2, guid="G-2", thread_originator_guid="OWN-1")
        reply = read(chat)[0][1]
        assert (reply.reply_to_from_me, reply.reply_to_text) == (False, None)

    def test_a_plain_message_refers_to_nothing(self, chat):
        chat.message(1, "Hi", chat=1, sender=1, guid="G-1")
        message = read(chat)[0][0]
        assert (message.reply_to, message.reply_to_from_me, message.reply_to_text, message.tapback) == ("", False, None, 0)

    @pytest.mark.parametrize("stored", ["p:0/OWN-1", "bp:OWN-1"], ids=["a part of the bubble", "the bubble whole"])
    def test_a_tapback_names_the_bubble_it_was_put_on(self, chat, stored):
        chat.message(1, "Dinner is at seven", chat=1, guid="OWN-1")
        chat.message(2, 'Loved "Dinner is at seven"', chat=1, sender=1, guid="G-2",
                     associated_message_type=2000, associated_message_guid=stored)
        tapback = read(chat)[0][1]
        assert tapback.tapback == 2000
        assert (tapback.reply_to, tapback.reply_to_from_me, tapback.reply_to_text) == ("OWN-1", True, "Dinner is at seven")

    def test_a_reply_whose_bubble_is_gone_still_names_its_guid(self, chat):
        chat.message(2, "and dessert?", chat=2, sender=1, guid="G-2", thread_originator_guid="DELETED")
        reply = read(chat)[0][0]
        assert (reply.reply_to, reply.reply_to_from_me, reply.reply_to_text) == ("DELETED", False, None)

    def test_a_row_in_no_chat_has_no_chat_guid(self, chat):
        chat.message(1, "orphan", sender=1, guid="G-1")
        message = read(chat)[0][0]
        assert (message.chat_guid, message.is_group, message.in_chat) == ("", False, False)


@pytest.mark.unit
class TestGroupChats:
    """Members per chat, the newest message as the activity, 1:1 chats left out."""

    def test_groups_come_with_named_members_and_their_latest_message(self, chat):
        chat.handle(3, "carol@example.com").chat(4, "any;+;chat4", [2, 3], name="", style=GROUP)
        chat.message(100, "a", ts=NOW - 50, chat=2, sender=1).message(101, "b", ts=NOW - 10, chat=2, sender=2)
        chat.message(102, "c", ts=NOW - 30, chat=4, sender=2)
        groups = {g.guid: g for g in MessagesDatabase(chat.path).group_chats(name)}
        assert set(groups) == {"any;+;chat2", "any;+;chat3", "any;+;chat4"}
        assert groups["any;+;chat2"] == GroupChat("any;+;chat2", "Family", ["Alice", "Bob"], apple(NOW - 10),
                                                  ["+46701740605", "+46701740606"])
        assert groups["any;+;chat4"].handles == ["+46701740606", "carol@example.com"], \
            "the handles are what a consumer matches its own list of people against"
        unnamed = groups["any;+;chat4"]
        assert unnamed.name is None and unnamed.members == ["Bob", "carol@example.com"]
        assert unnamed.label == "Bob, carol@example.com"
        assert groups["any;+;chat3"].last_active == 0, "a group nobody has written in yet"

    def test_a_group_is_described_by_its_name_and_members(self):
        assert GroupChat("any;+;chat1", "Family", ["Alice", "Bob"]).describe() == "Family (Alice, Bob)"
        assert GroupChat("any;+;chat1", None, ["Alice", "Bob"]).describe() == "Alice, Bob"

    def test_an_unreadable_db_yields_no_groups(self, tmp_path):
        assert MessagesDatabase(tmp_path / "missing.db").group_chats(name) == []
