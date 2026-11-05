"""The IMAP/SMTP implementation of MailAccount, parametrised by hosts, folders and a search hook.

`PRESETS` holds what a known service needs: iCloud's SMTP server files nothing, so a Sent copy is
appended over IMAP; Gmail files its own Sent copy and rejects non-ASCII SEARCH strings, so its
preset folds the query (`gmail_search_criteria`). An account built from a preset may still override
a host or port (Gmail's `email.smtp_host`).

An account signs in with a password, or — when it is given an `oauth_token` supplier — with
XOAUTH2 (Gmail through the Google integration): the SASL string both servers take is
`xoauth2_string`, which imap_tools builds itself for IMAP and `_send` hands to smtplib's
`AUTH XOAUTH2`."""
import smtplib
import time
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from email.message import Message
from typing import Any, Callable, List, Optional

from imap_tools import MailBox, AND, MailMessageFlags
from imap_tools.errors import MailboxFolderSelectError

from common.logging import trace, info, error
from .MailAccount import MailAccount, FolderNotFound

SMTP_TIMEOUT = 20
# imap_tools defaults to no socket timeout — Gmail IMAP has been observed hanging for minutes
# on connect/login, blocking the bg command worker. 60s is the per-recv socket timeout,
# generous enough for server-side SEARCH on a large mailbox while still bailing out on a hang.
IMAP_TIMEOUT = 60

# Letters NFKD cannot decompose to ASCII base + combining mark.
_FOLD_MAP = str.maketrans({
    'ø': 'o', 'Ø': 'O', 'æ': 'ae', 'Æ': 'AE', 'œ': 'oe', 'Œ': 'OE',
    'ß': 'ss', 'ð': 'd', 'Ð': 'D', 'þ': 'th', 'Þ': 'Th', 'đ': 'd', 'Đ': 'D',
    'ł': 'l', 'Ł': 'L',
})


def _ascii_fold(text: str) -> str:
    decomposed = unicodedata.normalize('NFKD', text.translate(_FOLD_MAP))
    return ''.join(ch for ch in decomposed if not unicodedata.combining(ch))


def xoauth2_string(address: str, access_token: str) -> str:
    """The SASL XOAUTH2 initial response as Gmail's IMAP and SMTP servers expect it (before
    the base64 the protocol wraps it in)."""
    return f"user={address}\x01auth=Bearer {access_token}\x01\x01"


def plain_search_criteria(search_string: str):
    return AND(text=search_string)


def gmail_search_criteria(search_string: str):
    """Gmail rejects any SEARCH whose command line contains raw 8-bit bytes ("BAD Could not
    parse command") — RFC 3501 requires non-ASCII strings to be sent as literals, which neither
    imap_tools nor imaplib does. Rather than drop to raw imaplib, fold the query to ASCII:
    Gmail's search index matches diacritic-folded terms (verified live — "Bjorkangen" finds
    "Björkängen")."""
    folded = _ascii_fold(search_string)
    if not folded.isascii():
        # Chars that don't fold (emoji, CJK, ...): drop them if enough of the query survives
        # to be meaningful; otherwise leave the query non-ASCII so Gmail's BAD reply surfaces
        # to the LLM instead of an everything-matches empty search.
        stripped = folded.encode('ascii', 'ignore').decode()
        if any(c.isalnum() for c in stripped):
            folded = stripped
    if folded != search_string:
        info(f"Gmail search: folded query '{search_string}' -> '{folded}'")
    escaped = folded.replace('\\', '\\\\').replace('"', '\\"')
    return AND(f'X-GM-RAW "{escaped}"')


@dataclass(frozen=True)
class ImapPreset:
    imap_host: str
    smtp_host: str
    folder: str = "INBOX"
    # Where a sent copy is filed over IMAP; None when the server does that itself.
    sent_folder: Optional[str] = None
    smtp_port: int = 587
    auth_help: str = "check the account password"
    search_criteria: Callable[[str], Any] = plain_search_criteria


PRESETS: dict[str, ImapPreset] = {
    "icloud": ImapPreset("imap.mail.me.com", "smtp.mail.me.com", sent_folder="Sent Messages",
                         auth_help="regenerate the app-specific password at appleid.apple.com"),
    # Gmail files SMTP sends in "[Gmail]/Sent Mail" itself; an IMAP copy would duplicate them.
    # Moving out of "[Gmail]/All Mail" trashes the message, so the assistant lists a specific
    # folder (e.g. "@SaneLater") before moving rather than moving from a search.
    "gmail": ImapPreset("imap.gmail.com", "smtp.gmail.com", folder="[Gmail]/All Mail",
                        auth_help="regenerate the app password at myaccount.google.com/apppasswords",
                        search_criteria=gmail_search_criteria),
}


class ImapMailAccount(MailAccount):
    """A mailbox reached over IMAP for reading and SMTP (STARTTLS, password login) for sending.

    `credentials` is read on every call, so a password set through config_set works without a
    rebuild; a KeyError from it means "not configured". With an `oauth_token` supplier the
    password is unused and every login is XOAUTH2 with the token it returns — a supplier that
    raises (not signed in) fails the call with its message. The hosts come from the preset
    unless given, and the search hook decides how a free-text query becomes IMAP criteria."""

    IMAP_TIMEOUT = IMAP_TIMEOUT

    def __init__(self, name: str, id_prefix: str, credentials: Callable[[], tuple[str, str]],
                 preset: str = "", *, imap_host: str = "", smtp_host: str = "", smtp_port: int = 0,
                 folder: str = "", sent_folder: Optional[str] = None, auth_help: str = "",
                 search_criteria: Optional[Callable[[str], Any]] = None,
                 oauth_token: Optional[Callable[[], str]] = None):
        super().__init__()
        base = PRESETS[preset] if preset else ImapPreset(imap_host, smtp_host)
        self._name = name
        self._id_prefix = id_prefix
        self._credentials = credentials
        self.preset = preset
        self.imap_host = imap_host or base.imap_host
        self.smtp_host = smtp_host or base.smtp_host
        self.smtp_port = int(smtp_port or base.smtp_port)
        self.default_folder = folder or base.folder
        self.sent_folder = sent_folder if sent_folder is not None else base.sent_folder
        self.AUTH_HELP = auth_help or base.auth_help
        self._search_criteria = search_criteria or base.search_criteria
        self._oauth_token = oauth_token
        if not self.imap_host or not self.smtp_host:
            raise ValueError(f"Mail account '{name}': an IMAP and an SMTP host are needed (or a preset)")

    @property
    def name(self) -> str:
        return self._name

    @property
    def id_prefix(self) -> str:
        return self._id_prefix

    @property
    def address(self) -> str:
        try:
            return self._credentials()[0] or ""
        except KeyError as e:
            trace(f"{self.name}: address not configured ({e})")
            return ""

    def _mailbox(self, folder: Optional[str] = None):
        address, password = self._credentials()
        mailbox = MailBox(self.imap_host, timeout=self.IMAP_TIMEOUT)
        folder_kwargs = {} if folder is None else {"initial_folder": folder}
        if self._oauth_token is not None:
            return mailbox.xoauth2(address, self._oauth_token(), **folder_kwargs)
        return mailbox.login(address, password, **folder_kwargs)

    def _timed_fetch(self, mailbox, description, **kwargs):
        trace(f"  {self.name}: IMAP query ({description})...")
        start = time.time()
        messages = list(mailbox.fetch(**kwargs))
        trace(f"  {self.name}: got {len(messages)} results in {time.time() - start:.1f}s")
        return messages

    # A listing is bounded to this many days first: unbounded, `reverse=True` makes the server
    # search the whole folder, and on Gmail's "[Gmail]/All Mail" (the entire account) two such
    # listings hung 5.7 minutes before the socket timed out (2026-11-04). A folder with fewer
    # than MAX_RESULTS messages in the window is listed whole — such a folder is small.
    LISTING_DAYS = 30

    def _list(self, folder: str, unread_only: bool) -> list:
        fetch_kwargs = dict(limit=self.MAX_RESULTS, reverse=True, mark_seen=False, headers_only=True)
        since = date.today() - timedelta(days=self.LISTING_DAYS)
        try:
            with self._mailbox(folder) as mailbox:
                recent = self._timed_fetch(mailbox, f"fetch latest since {since}",
                                           criteria=self._listing_criteria(unread_only, since), **fetch_kwargs)
                if len(recent) >= self.MAX_RESULTS:
                    return recent
                trace(f"  {self.name}: {len(recent)} message(s) in the last {self.LISTING_DAYS} days, "
                      f"listing the whole folder")
                return self._timed_fetch(mailbox, "fetch latest",
                                         criteria=self._listing_criteria(unread_only, None), **fetch_kwargs)
        except MailboxFolderSelectError as e:
            raise FolderNotFound(str(e)) from e

    @staticmethod
    def _listing_criteria(unread_only: bool, since: Optional[date]):
        terms = {}
        if unread_only:
            terms["seen"] = False
        if since is not None:
            terms["date_gte"] = since
        return AND(**terms) if terms else "ALL"

    def _search(self, search_string: str) -> list:
        with self._mailbox(self.default_folder) as mailbox:
            # No date filter: server-side SEARCH is index-backed (Gmail) or fast enough
            # (iCloud), and the fetch is capped at MAX_RESULTS newest matches anyway.
            # headers_only keeps the capped fetch cheap — the list view never needs bodies.
            return self._timed_fetch(mailbox, f"search '{search_string}'",
                                     criteria=self._search_criteria(search_string), charset='UTF-8',
                                     limit=self.MAX_RESULTS, reverse=True, mark_seen=False, headers_only=True)

    def _fetch(self, uid: str, folder: str):
        with self._mailbox(folder) as mailbox:
            messages = self._timed_fetch(mailbox, f"read UID {uid}", criteria=AND(uid=uid), mark_seen=False)
        return messages[0] if messages else None

    def _folders(self) -> List[str]:
        with self._mailbox() as mailbox:
            return [f.name for f in mailbox.folder.list()]

    def _move(self, uid: str, source: str, dest_folder: str) -> None:
        # On Gmail a move is a relabelling — the source label is removed and the
        # destination added — so it is reversible.
        try:
            with self._mailbox(source) as mailbox:
                mailbox.move(uid, dest_folder)
        except MailboxFolderSelectError as e:
            raise FolderNotFound(str(e)) from e

    def _recent_from(self, sender_contains: str, limit: int) -> list:
        since_date = date.today() - timedelta(days=1)
        with self._mailbox(self.default_folder) as mailbox:
            return self._timed_fetch(mailbox, f"raw fetch sender~'{sender_contains}'",
                                     criteria=AND(from_=sender_contains, date_gte=since_date),
                                     limit=limit, reverse=True, mark_seen=False)

    def _send(self, msg: Message, recipients: List[str]) -> str:
        try:
            address, password = self._credentials()
        except KeyError as e:
            error(f"Email send: {self.name} credentials not configured: {e}")
            return f"{self.name} credentials not configured"
        if not address or (not password and self._oauth_token is None):
            error(f"Email send: {self.name} credentials not configured")
            return f"{self.name} credentials not configured"
        try:
            token = self._oauth_token() if self._oauth_token is not None else ""
        except Exception as e:
            error(f"Email send: {self.name} has no OAuth token: {e}")
            return f"{self.name}: {e}"
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=SMTP_TIMEOUT) as server:
                server.starttls()
                if self._oauth_token is not None:
                    server.ehlo_or_helo_if_needed()
                    server.auth("XOAUTH2", lambda challenge=None: xoauth2_string(address, token))
                else:
                    server.login(address, password)
                server.sendmail(address, list(recipients), msg.as_string())
        except smtplib.SMTPAuthenticationError as e:
            error(f"Email send auth failed: {e}")
            return f"{self.name} SMTP auth failed — {self.AUTH_HELP}"
        except smtplib.SMTPRecipientsRefused as e:
            error(f"Email send: recipient refused: {e}")
            return f"recipient refused: {', '.join(recipients)}"
        except smtplib.SMTPException as e:
            error(f"Email send SMTP error: {e}")
            return f"SMTP error: {e}"
        except (TimeoutError, OSError) as e:
            error(f"Email send network error: {e}")
            return f"network error sending email: {e}"
        self._save_to_sent(msg)
        return ""

    def _save_to_sent(self, msg: Message) -> bool:
        """A failed Sent copy is logged but is not a failure: the mail is already out."""
        if not self.sent_folder:
            trace(f"{self.name}: the server files sent mail itself — no Sent copy to save")
            return True
        try:
            with self._mailbox() as mailbox:
                mailbox.append(msg.as_bytes(), self.sent_folder, flag_set=[MailMessageFlags.SEEN])
            info(f"{self.name}: copy of {msg.get('Subject', '')!r} filed in '{self.sent_folder}'")
            return True
        except Exception as e:
            error(f"{self.name}: the mail went out but filing a copy in '{self.sent_folder}' failed: {e}")
            return False

    def check(self) -> bool:
        with self._mailbox(self.default_folder):
            pass
        return True
