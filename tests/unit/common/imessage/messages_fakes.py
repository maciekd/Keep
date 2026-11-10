"""A chat.db and an AddressBook of Apple's shape on disk, built row by row: the Messages host
layer is tested over these files, never over the Mac's own."""
import sqlite3
import time

from common.imessage.MessagesDatabase import APPLE_EPOCH_OFFSET

NOW = time.time()
DIRECT, GROUP = 45, 43      # chat.style


def apple(unix_ts: float) -> int:
    return int((unix_ts - APPLE_EPOCH_OFFSET) * 1_000_000_000)


class ChatDb:
    """Writes a chat.db; `send_later_columns=False` gives the schema of a Mac before macOS 15."""

    def __init__(self, path, send_later_columns: bool = True):
        self.path = str(path)
        schedule = ", schedule_type INTEGER DEFAULT 0, schedule_state INTEGER DEFAULT 0" if send_later_columns else ""
        self._run(f"""
            CREATE TABLE message (ROWID INTEGER PRIMARY KEY AUTOINCREMENT, guid TEXT, text TEXT, attributedBody BLOB, date INTEGER,
                                  is_from_me INTEGER DEFAULT 0, handle_id INTEGER DEFAULT 0,
                                  is_system_message INTEGER DEFAULT 0, service TEXT,
                                  thread_originator_guid TEXT, associated_message_guid TEXT,
                                  associated_message_type INTEGER DEFAULT 0{schedule});
            CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT, service TEXT);
            CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT, style INTEGER, display_name TEXT);
            CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
            CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER);
            CREATE TABLE attachment (ROWID INTEGER PRIMARY KEY, filename TEXT, mime_type TEXT);
            CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER);
        """)

    def _run(self, script: str, *statements):
        conn = sqlite3.connect(self.path)
        conn.executescript(script)
        for sql, params in statements:
            conn.execute(sql, params)
        conn.commit()
        conn.close()

    def execute(self, sql: str, params=()):
        self._run("", (sql, params))
        return self

    def handle(self, rowid: int, handle_id, service: str = "iMessage"):
        return self.execute("INSERT INTO handle VALUES (?, ?, ?)", (rowid, handle_id, service))

    def chat(self, rowid: int, guid: str, members=(), name=None, style: int = DIRECT):
        self.execute("INSERT INTO chat VALUES (?, ?, ?, ?)", (rowid, guid, style, name))
        for member in members:
            self.execute("INSERT INTO chat_handle_join VALUES (?, ?)", (rowid, member))
        return self

    def message(self, rowid: int, text, ts: float = NOW, chat=None, sender=None, system: bool = False, **columns):
        """An incoming message from the handle row `sender`, or — without one — an outgoing
        message, which Apple stores with handle_id = 0. `columns` sets any other column."""
        values = {"ROWID": rowid, "text": text, "date": apple(ts), "is_from_me": int(sender is None),
                  "handle_id": sender or 0, "is_system_message": int(system), **columns}
        self.execute(f"INSERT INTO message ({', '.join(values)}) VALUES ({', '.join('?' * len(values))})",
                     tuple(values.values()))
        if chat is not None:
            self.execute("INSERT INTO chat_message_join VALUES (?, ?)", (chat, rowid))
        return self

    def attachment(self, rowid: int, message: int, filename, mime_type):
        self.execute("INSERT INTO attachment VALUES (?, ?, ?)", (rowid, filename, mime_type))
        return self.execute("INSERT INTO message_attachment_join VALUES (?, ?)", (message, rowid))


def address_book(root, people):
    """Write a minimal AddressBook-v22.abcddb under `root` and return the AddressBook folder;
    `people` are (first, last, phones, emails) — or (first, middle, last, phones, emails) — with
    phones as (number, label) tuples."""
    folder = root / "AddressBook"
    db = folder / "Sources" / "ABC" / "AddressBook-v22.abcddb"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE ZABCDRECORD (Z_PK INTEGER PRIMARY KEY, ZFIRSTNAME TEXT, ZMIDDLENAME TEXT, "
                     "ZLASTNAME TEXT)")
        conn.execute("CREATE TABLE ZABCDPHONENUMBER (Z_PK INTEGER PRIMARY KEY, ZOWNER INTEGER, "
                     "ZFULLNUMBER TEXT, ZLABEL TEXT)")
        conn.execute("CREATE TABLE ZABCDEMAILADDRESS (Z_PK INTEGER PRIMARY KEY, ZOWNER INTEGER, ZADDRESS TEXT)")
        for pk, person in enumerate(people, start=1):
            first, middle, last, phones, emails = person if len(person) == 5 else (person[0], None, *person[1:])
            conn.execute("INSERT INTO ZABCDRECORD VALUES (?, ?, ?, ?)", (pk, first, middle, last))
            for number, label in phones:
                conn.execute("INSERT INTO ZABCDPHONENUMBER (ZOWNER, ZFULLNUMBER, ZLABEL) VALUES (?, ?, ?)",
                             (pk, number, label))
            for email in emails:
                conn.execute("INSERT INTO ZABCDEMAILADDRESS (ZOWNER, ZADDRESS) VALUES (?, ?)", (pk, email))
    return folder
