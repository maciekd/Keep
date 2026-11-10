"""The iMessage catch-up on a chat.db whose ROWIDs are not in date order: Messages in iCloud
downloads a fresh library's history in chunks (newest first, old, recent again), so the last
ROWID from before the horizon can sit near the end of the table. The cursor must start just
before the first in-horizon row and the reads must skip the pre-horizon rows between."""
import sqlite3
import threading
import time
from unittest.mock import Mock

import pytest

from assistant.integration.FeedRunner import FeedRunner
from common.FloxyDatabase import FloxyDatabase
from common.imessage import Hosts

from common.imessage.Contacts import Contacts
from common.imessage.LocalMessagesHost import LocalMessagesHost
from common.imessage.MessagesDatabase import MessagesDatabase, APPLE_EPOCH_OFFSET
from integrations.imessage.IMessageIntegration import IMessageIntegration

NOW = time.time()
HORIZON = NOW - 90 * 86400
DAY = 86400


def apple(unix_ts: float) -> int:
    return int((unix_ts - APPLE_EPOCH_OFFSET) * 1_000_000_000)


def make_chat_db(path, rows):
    """`rows`: (rowid, unix_ts, text) incoming messages from one handle; the tables the
    feed's queries join, nothing more."""
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE message (ROWID INTEGER PRIMARY KEY AUTOINCREMENT, guid TEXT, text TEXT, attributedBody BLOB, date INTEGER,
                                  is_from_me INTEGER, handle_id INTEGER, is_system_message INTEGER DEFAULT 0,
                                  service TEXT, thread_originator_guid TEXT, associated_message_guid TEXT,
                                  associated_message_type INTEGER DEFAULT 0,
                                  schedule_type INTEGER DEFAULT 0, schedule_state INTEGER DEFAULT 0);
            CREATE INDEX message_idx_date ON message(date);
            CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT, service TEXT);
            CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT, style INTEGER, display_name TEXT);
            CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
            CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER);
            CREATE TABLE attachment (ROWID INTEGER PRIMARY KEY, filename TEXT, mime_type TEXT);
            CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER);
            INSERT INTO handle VALUES (1, '+46701740605', 'iMessage');
            INSERT INTO chat VALUES (1, 'iMessage;-;+46701740605', 45, NULL);
        """)
        for rowid, ts, text in rows:
            conn.execute("INSERT INTO message (ROWID, text, date, is_from_me, handle_id) VALUES (?, ?, ?, 0, 1)",
                         (rowid, text, apple(ts)))
            conn.execute("INSERT INTO chat_message_join VALUES (1, ?)", (rowid,))


# ROWID order is nothing like date order: a recent chunk first, then years-old ones, then
# recent again, and an old chunk at the very end — the rows a fresh library syncs down.
ROWS = [
    (1, NOW - 10 * DAY, "recent, first chunk"),
    (2, NOW - 9 * DAY, "recent, first chunk 2"),
    (3, HORIZON - 400 * DAY, "old"),
    (4, HORIZON - 399 * DAY, "old 2"),
    (5, HORIZON + 5 * DAY, "just inside the horizon"),
    (6, HORIZON - 1 * DAY, "just before the horizon"),
    (7, NOW - 2 * DAY, "recent again"),
    (8, HORIZON - 2000 * DAY, "old, written last"),
]
IN_HORIZON = {text for _, ts, text in ROWS if ts >= HORIZON}


@pytest.fixture
def feed(tmp_path):
    db = tmp_path / "chat.db"
    make_chat_db(db, ROWS)
    IMessageIntegration.reset_instance()
    database = MessagesDatabase(db)
    contacts = Contacts(database, tmp_path / "AddressBook")
    contacts._contact_cache = {"+46701740605": "Alice"}
    contacts._contact_cache_time = 1e18
    m = IMessageIntegration()
    m.host = LocalMessagesHost(database, contacts)
    yield m, db
    IMessageIntegration.reset_instance()


def catch_up(m, cursor):
    texts = []
    while True:
        batch = m.fetch(cursor, backfill=True)
        texts += [e.content.split(": ", 1)[1] for e in batch.entries]
        cursor = batch.cursor
        if batch.exhausted:
            return texts, cursor


@pytest.mark.unit
class TestOutOfOrderRowids:
    def test_the_catch_up_reads_every_in_horizon_row_and_no_older_one(self, feed):
        m, _ = feed
        cursor = m.initial_cursor(HORIZON)
        assert cursor == "0"    # ROWID 1 is inside the horizon, so the catch-up starts before it
        texts, cursor = catch_up(m, cursor)
        assert set(texts) == IN_HORIZON
        assert cursor == "7"    # the old row written last is never read; the date test keeps it out of every poll

    def test_remaining_counts_what_fetch_will_read(self, feed):
        m, _ = feed
        cursor = m.initial_cursor(HORIZON)
        assert m.remaining(cursor) == len(IN_HORIZON)
        _, cursor = catch_up(m, cursor)
        assert m.remaining(cursor) == 0

    def test_an_old_row_synced_down_late_is_not_a_fresh_message(self, feed):
        m, db = feed
        _, cursor = catch_up(m, m.initial_cursor(HORIZON))
        with sqlite3.connect(db) as conn:
            conn.execute("INSERT INTO message (ROWID, text, date, is_from_me, handle_id) VALUES (9, ?, ?, 0, 1)",
                         ("a 2013 message iCloud wrote today", apple(HORIZON - 3000 * DAY)))
            conn.execute("INSERT INTO message (ROWID, text, date, is_from_me, handle_id) VALUES (10, ?, ?, 0, 1)",
                         ("a real new message", apple(NOW)))
            conn.execute("INSERT INTO chat_message_join VALUES (1, 9), (1, 10)")
        batch = m.fetch(cursor, backfill=False)
        assert [e.content.split(": ", 1)[1] for e in batch.entries] == ["a real new message"]
        assert batch.cursor == "10"

    def test_a_horizon_past_every_row_starts_at_the_end_of_the_table(self, feed):
        m, _ = feed
        assert m.initial_cursor(NOW + DAY) == "8"
        assert m.remaining("8") == 0

    def test_without_a_horizon_every_row_is_read(self, feed):
        m, _ = feed
        m.horizon_ts = None
        texts, _ = catch_up(m, "0")
        assert len(texts) == len(ROWS)


SATELLITE = "http://mini.local:8898"


@pytest.fixture
def floxy_db(tmp_path):
    db = FloxyDatabase.__new__(FloxyDatabase)
    db.db_path = str(tmp_path / "floxy_test.db")
    db.lock = threading.Lock()
    db._init_database()
    return db


@pytest.mark.unit
class TestTheCursorBelongsToOneAccount:
    """A ROWID is a place in one chat.db. The feed's row outlives a change of `imessage.host`,
    and a ROWID of the database read before says nothing about the one read now."""

    def resume(self, m, floxy_db, account=Hosts.LOCAL, **stored):
        """The runner's start over the stored row, with the integration reading `account`: (the
        cursor it goes on from, the texts a catch-up from there reads, the row as kept)."""
        m.cursor_source = account
        floxy_db.save_feed("iMessage", horizon=HORIZON, generation=m.cursor_generation, **stored)
        runner = FeedRunner(m, Mock(), HORIZON, db=floxy_db)
        runner._load_cursor()
        texts, _ = catch_up(m, runner._cursor)
        return runner._cursor, set(texts), floxy_db.get_feed("iMessage")

    def test_a_cursor_of_this_macs_messages_is_not_applied_to_a_satellites(self, feed, floxy_db):
        """Past the end of the smaller database the feed would be live, healthy and read nothing."""
        m, _ = feed
        cursor, texts, row = self.resume(m, floxy_db, SATELLITE, cursor="250000", source=Hosts.LOCAL)
        assert cursor == "0" and texts == IN_HORIZON
        assert row["source"] == SATELLITE, "the next start resumes: the cursor is this account's now"

    def test_nor_one_below_the_other_databases_end(self, feed, floxy_db):
        """Inside the other database the old ROWID would silently skip what lies before it."""
        m, _ = feed
        cursor, texts, _ = self.resume(m, floxy_db, Hosts.LOCAL, cursor="6", source=SATELLITE)
        assert cursor == "0" and texts == IN_HORIZON

    def test_an_install_from_before_the_account_was_kept_goes_on_where_it_was(self, feed, floxy_db):
        """Upgrading must not read five years of history again: such a row is this Mac's own Messages."""
        m, _ = feed
        cursor, texts, row = self.resume(m, floxy_db, Hosts.LOCAL, cursor="6")
        assert cursor == "6" and texts == {"recent again"}
        assert row["source"] == Hosts.LOCAL

    def test_and_starts_over_when_it_was_pointed_at_a_satellite_since(self, feed, floxy_db):
        m, _ = feed
        cursor, texts, _ = self.resume(m, floxy_db, SATELLITE, cursor="6")
        assert cursor == "0" and texts == IN_HORIZON

    def test_a_cursor_past_the_last_row_is_of_another_database_whatever_the_account_says(self, feed, floxy_db):
        """A satellite reinstalled under the same address, a restored Mac: the account is spelled the same."""
        m, _ = feed
        cursor, texts, _ = self.resume(m, floxy_db, SATELLITE, cursor="250000", source=SATELLITE)
        assert cursor == "0" and texts == IN_HORIZON

    def test_deleting_the_newest_messages_does_not_send_the_feed_back_to_the_horizon(self, feed, floxy_db):
        """The cursor stood on a message the user deleted before a restart: still this database,
        so nothing is read again — a catch-up over five years for a deleted text would be the bug."""
        import sqlite3
        m, _ = feed
        conn = sqlite3.connect(m.host.database.path)
        conn.execute("DELETE FROM message WHERE ROWID >= 7")
        conn.commit()
        conn.close()
        cursor, texts, _ = self.resume(m, floxy_db, Hosts.LOCAL, cursor="8", source=Hosts.LOCAL)
        assert cursor == "8" and texts == set()

    def test_a_row_dated_centuries_ahead_does_not_end_the_table_before_it(self, feed, floxy_db):
        """The end of the table is its counter, not "the row before the first one dated after
        2100": with a garbage date on an early row the cursor would be "past the last row" at
        every start, and five years read again each time."""
        import sqlite3
        m, _ = feed
        conn = sqlite3.connect(m.host.database.path)
        conn.execute("UPDATE message SET date = ? WHERE ROWID = 3", (apple(NOW + 200 * 365 * DAY),))
        conn.commit()
        conn.close()
        cursor, _, _ = self.resume(m, floxy_db, Hosts.LOCAL, cursor="8", source=Hosts.LOCAL)
        assert cursor == "8"

    def test_the_same_account_resumes_also_at_the_very_end(self, feed, floxy_db):
        m, _ = feed
        cursor, texts, _ = self.resume(m, floxy_db, SATELLITE, cursor="8", source=SATELLITE)
        assert cursor == "8" and texts == set()
