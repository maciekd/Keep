"""The mail account contract: one mailbox the assistant lists, searches, reads, moves and sends from.

Everything that is not the wire protocol lives here — the short ids the LLM sees (prefix +
counter, cached per listing), the listing and body rendering, the attachment handling (PDF text,
media description), send validation with its rate limit and the subject lookup the progress
notifications use. A backend (IMAP/SMTP today, Graph later) implements the handful of `_`
primitives at the bottom of the class and nothing else.

A message is anything with the attributes of imap_tools' `MailMessage` the renderers read
(`uid`, `date`, `from_`, `to`, `subject`, `flags`, `text`, `html`, `attachments` with `filename`,
`content_type`, `payload`, `size`, `content_disposition`, `content_id`); a backend that is not
IMAP adapts its own shape to it."""
import re
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from email.message import Message
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid
from typing import Dict, List, Optional

from common.logging import trace, info, error
from common.RateLimiter import RateLimiter
from assistant.tools import PdfExtractor
from assistant.tools.MediaAnalyzer import media_analyzer, media_kind, is_image_mime, is_video_mime, KIND_VIDEO


MIN_MEDIA_SIZE = 10 * 1024
SEND_RATE_LIMIT_WINDOW = 60.0
SEND_RATE_LIMIT_MAX = 5
EMAIL_ADDRESS_PATTERN = re.compile(r"^[^@\s,]+@[^@\s,]+\.[^@\s,]+$")


class FolderNotFound(Exception):
    """The backend has no folder by the name a listing or a move named."""


# Types a sender's client uses when it does not know (or care) what the file is; the extension
# then says more than the type does.
GENERIC_CONTENT_TYPES = ("", "application/octet-stream", "binary/octet-stream",
                         "application/x-download", "application/force-download")
KIND_PDF = "pdf"


def _attachment_kind(att) -> Optional[str]:
    """"pdf", "image", "video" or None. Trusts the MIME type; a generic one (octet-stream, empty)
    falls back on the filename extension."""
    content_type = (att.content_type or "").lower()
    if content_type == "application/pdf":
        return KIND_PDF
    if is_image_mime(content_type) or is_video_mime(content_type):
        return media_kind(content_type, att.filename)
    if content_type in GENERIC_CONTENT_TYPES:
        if (att.filename or "").lower().endswith(".pdf"):
            return KIND_PDF
        return media_kind(None, att.filename)
    return None


def _is_junk_media(att) -> bool:
    """Tiny files and inline (cid-referenced) images are signatures, logos and tracking pixels;
    anything Floxy cannot describe (svg, audio, ...) is skipped too."""
    size = att.size or 0
    if size < MIN_MEDIA_SIZE:
        return True
    if att.content_disposition == "inline" and att.content_id:
        return True
    return media_kind(att.content_type or "", att.filename) is None


def _analyze_media(payload: bytes, content_type: str, filename: str) -> str:
    label = "Video" if media_kind(content_type, filename) == KIND_VIDEO else "Image"
    return media_analyzer().describe_bytes(
        payload, mime_type=content_type, filename=filename,
        context=f"{label} attached to an email, filename: {filename}", fallback=f"[{label}]")


class MailAccount(ABC):
    """One mailbox as the assistant sees it; the protocol is the subclass's business.

    Emails are exposed to the LLM through short ids (prefix + counter) that are cached per
    listing; IMAP UIDs are folder-scoped, so the cache remembers its source folder and the
    per-email operations re-select it.
    """

    MAX_RESULTS = 35
    MAX_BODY_LENGTH = 8000
    # The folder a listing shows when the LLM names none.
    default_folder: str = "INBOX"
    # The preset the account was built from ("icloud", "gmail"), "" for a hand-configured one.
    preset: str = ""
    AUTH_HELP = "check the account password"

    def __init__(self):
        self._email_id_cache: Dict[str, str] = {}
        # Subjects keyed by the same short id, populated alongside the id-cache so
        # the read-email progress notification can name the email instead of
        # leaking the internal short id. Kept separate from _email_id_cache so the
        # UID-resolution path (_resolve_email_id/move) stays untouched.
        self._email_subject_cache: Dict[str, str] = {}
        self._next_cache_id: int = 0
        # The folder the current id-cache was built from. IMAP UIDs are
        # folder-scoped, so move() must re-select this exact folder to resolve a
        # short id back to a live UID. Defaults to the standard listing folder.
        self._cache_folder: Optional[str] = None
        # Listings can run concurrently (background commands of one reply run in
        # parallel); rebuilding the id-cache is a multi-step write, so it happens
        # under this lock and the readers take it too. The last listing to finish wins.
        self._cache_lock = threading.Lock()
        self._rate_limiter = RateLimiter(SEND_RATE_LIMIT_MAX, SEND_RATE_LIMIT_WINDOW)

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def id_prefix(self) -> str: ...

    @property
    @abstractmethod
    def address(self) -> str:
        """The account's own address: the From of what it sends, "" when not configured."""
        ...

    # --- the backend's primitives ---------------------------------------------------------

    @abstractmethod
    def _list(self, folder: str, unread_only: bool) -> list:
        """The newest MAX_RESULTS messages of `folder`, headers only; raises FolderNotFound."""
        ...

    @abstractmethod
    def _search(self, search_string: str) -> list:
        """The newest MAX_RESULTS matches in the default folder, headers only."""
        ...

    @abstractmethod
    def _fetch(self, uid: str, folder: str):
        """The whole message `uid` of `folder`, or None."""
        ...

    @abstractmethod
    def _folders(self) -> List[str]: ...

    @abstractmethod
    def _move(self, uid: str, source: str, dest_folder: str) -> None:
        """Raises FolderNotFound for a folder the backend does not have."""
        ...

    @abstractmethod
    def _recent_from(self, sender_contains: str, limit: int) -> list:
        """Up to `limit` of the newest messages of the last day whose sender contains the text."""
        ...

    @abstractmethod
    def _send(self, msg: Message, recipients: List[str]) -> str:
        """Hands a complete message to the server, filing a Sent copy when the server does not;
        "" on success or the short error the LLM reads."""
        ...

    @abstractmethod
    def check(self) -> bool:
        """The health check: signs in and returns True; may raise."""
        ...

    # --- sending -----------------------------------------------------------------------------

    def send(self, to: str, subject: str, body: str) -> str:
        """Sends a plain-text email. Returns "" on success or a short error string on
        failure (the convention the LLM relies on to detect and surface send failures)."""
        recipient = (to or "").strip()
        subject = (subject or "").strip()
        body = (body or "").strip()
        if not recipient:
            info("Email send skipped: empty recipient")
            return "empty recipient"
        if not subject:
            info("Email send skipped: empty subject")
            return "empty subject"
        if not body:
            info("Email send skipped: empty body")
            return "empty body"
        if not EMAIL_ADDRESS_PATTERN.match(recipient):
            info(f"Email send blocked: invalid recipient '{recipient}'")
            return f"invalid recipient: {recipient}"

        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["To"] = recipient
        return self.send_message(msg, [recipient])

    def send_message(self, msg: Message, recipients: List[str]) -> str:
        """Sends a prepared message. Fills in From, Date and Message-ID when the caller left
        them out. Returns "" on success or a short error string."""
        rate_err = self._rate_limiter.check()
        if rate_err:
            info(f"Email send blocked: {rate_err}")
            return rate_err

        address = self.address
        if not address:
            error(f"Email send: {self.name} credentials not configured")
            return f"{self.name} credentials not configured"

        if "From" not in msg:
            msg["From"] = address
        if "Date" not in msg:
            msg["Date"] = formatdate(localtime=True)
        if "Message-ID" not in msg:
            msg["Message-ID"] = make_msgid(domain=address.rsplit("@", 1)[-1])

        err = self._send(msg, list(recipients))
        if err:
            return err
        self._rate_limiter.record()
        info(f"Email sent from {self.name} to {', '.join(recipients)}: subject={msg.get('Subject', '')!r}")
        return ""

    # --- listing, reading, moving -----------------------------------------------------------

    def fetch_latest(self, folder=None, unread_only=False) -> str:
        # `folder`, when a non-empty string, names a folder/label to list (e.g. "@SaneLater")
        # instead of the default mailbox, so the assistant can browse a specific folder
        # before moving messages out of it. `unread_only` restricts the listing to unseen
        # messages — handy for sweeping a backlog without re-listing what is dealt with.
        folder = (str(folder).strip() if folder else "") or self.default_folder
        try:
            scope = "unread " if unread_only else ""
            info(f"Fetching latest {scope}emails from {self.name} ({folder})...")
            messages = self._list(folder, unread_only)
            self._cache_folder = folder
            return self._format_email_list(messages)
        except FolderNotFound:
            info(f"{self.name}: folder '{folder}' does not exist here — no emails.")
            return "No emails found."
        except Exception as e:
            error(f"Failed to fetch latest {self.name} emails: {e}")
            return f"Error fetching {self.name} emails: {e}"

    def search(self, search_string) -> str:
        try:
            info(f"Searching {self.name} for: {search_string}")
            messages = self._search(search_string)
            self._cache_folder = self.default_folder
            return self._format_email_list(messages)
        except Exception as e:
            error(f"Failed to search {self.name} emails: {e}")
            return f"Error searching {self.name} emails: {e}"

    def read(self, email_id) -> str:
        try:
            email_id = str(email_id).strip()
            email_id = self._resolve_email_id(email_id)
            # UIDs are scoped to the folder the id-cache was listed from (see move()),
            # so select that folder — reading from INBOX after listing @SaneLater would
            # resolve the UID against the wrong mailbox.
            folder = self._cache_folder or self.default_folder
            info(f"Reading {self.name} email UID: {email_id} from '{folder}'")
            message = self._fetch(email_id, folder)
            if message is None:
                return f"Email with ID {email_id} not found."
            return self._format_email_body(message)
        except Exception as e:
            error(f"Failed to read {self.name} email {email_id}: {e}")
            return f"Error reading email: {e}"

    def list_folders(self, params=None) -> str:
        """List every folder/label so the assistant can pick a valid destination for
        move() instead of guessing label names."""
        try:
            info(f"Listing {self.name} folders...")
            folders = self._folders()
            return "\n".join(folders) if folders else "No folders found."
        except Exception as e:
            error(f"Failed to list {self.name} folders: {e}")
            return f"Error listing {self.name} folders: {e}"

    def move(self, email_id, dest_folder) -> str:
        """Move a previously-listed email to another folder/label. The short id is resolved
        against `self._cache_folder`, the folder the id-cache was built from."""
        try:
            email_id = str(email_id).strip()
            dest_folder = str(dest_folder).strip().strip('"').strip("'")
            uid = self._resolve_email_id(email_id)
            source = self._cache_folder or self.default_folder
            info(f"Moving {self.name} email UID {uid}: '{source}' -> '{dest_folder}'")
            self._move(uid, source, dest_folder)
            # Echo the subject back into the result so the model's feedback channel
            # carries the human-meaningful identity, not just the opaque short id —
            # otherwise it can't tell which message it actually moved. Truncated the
            # same way as the Telegram progress notification.
            subject = self.subject_for(email_id)
            if subject and subject.strip() and subject.strip() != "(no subject)":
                s = subject.strip()
                if len(s) > 50:
                    s = s[:50].rstrip() + "…"
                return f'Moved email {email_id} ("{s}") to {dest_folder}.'
            return f"Moved email {email_id} to {dest_folder}."
        except FolderNotFound as e:
            error(f"Failed to move {self.name} email {email_id}: folder error: {e}")
            return f"Error moving email: folder not found ({e})"
        except Exception as e:
            error(f"Failed to move {self.name} email {email_id}: {e}")
            return f"Error moving email: {e}"

    def fetch_recent_raw(self, sender_contains: str, since_minutes: int = 10, limit: int = 5) -> List:
        """Recent raw messages from a sender, the last `since_minutes` only (the 2FA mail polling)."""
        try:
            messages = self._recent_from(sender_contains, limit)
            if not messages:
                trace(f"{self.name}: no recent mail from a sender matching '{sender_contains}'")
                return []
            tz = messages[0].date.tzinfo if messages[0].date else None
            cutoff = datetime.now(tz) - timedelta(minutes=since_minutes)
            return [m for m in messages if m.date and m.date >= cutoff]
        except Exception as e:
            error(f"Failed to fetch raw {self.name} emails for sender '{sender_contains}': {e}")
            return []

    # --- the short-id cache -----------------------------------------------------------------

    def _generate_short_id(self) -> str:
        id_num = self._next_cache_id
        self._next_cache_id = (self._next_cache_id + 1) % (26 * 26)
        first = chr(65 + (id_num // 26))
        second = chr(65 + (id_num % 26))
        return f"{first}{second}"

    def _resolve_email_id(self, email_id: str) -> str:
        prefix = f"{self.id_prefix}-"
        if email_id.startswith(prefix):
            email_id = email_id[len(prefix):]
        with self._cache_lock:
            return self._email_id_cache.get(email_id, email_id)

    def subject_for(self, email_id: str) -> Optional[str]:
        """Cached subject for a short id from the most recent listing, or None on
        a miss (id never listed, cache cleared since). Used only for the friendly
        read-email notification, so a miss is harmless."""
        short_id = str(email_id).strip()
        prefix = f"{self.id_prefix}-"
        if short_id.startswith(prefix):
            short_id = short_id[len(prefix):]
        with self._cache_lock:
            return self._email_subject_cache.get(short_id)

    # --- rendering --------------------------------------------------------------------------

    @staticmethod
    def _format_date(dt: datetime) -> str:
        if not dt:
            return "unknown"
        # Header dates are tz-aware (usually UTC or the sender's tz); show them
        # in the server's local time so the LLM reasons in the user's clock.
        if dt.tzinfo is not None:
            dt = dt.astimezone()
        return dt.strftime("%b %d %H:%M")

    def _format_email_list(self, messages: list) -> str:
        if not messages:
            return "No emails found."
        lines = []
        with self._cache_lock:
            self._email_id_cache.clear()
            self._email_subject_cache.clear()
            self._next_cache_id = 0
            for msg in messages:
                short_id = self._generate_short_id()
                self._email_id_cache[short_id] = msg.uid
                date_str = self._format_date(msg.date)
                from_addr = msg.from_ or "unknown"
                subject = msg.subject or "(no subject)"
                self._email_subject_cache[short_id] = subject
                unread = " [UNREAD]" if "\\Seen" not in (msg.flags or ()) else ""
                lines.append(f"[{self.id_prefix}-{short_id}]{unread} {date_str} | From: {from_addr} | Subject: {subject}")
        return "\n".join(lines)

    @staticmethod
    def strip_html(html: str) -> str:
        from bs4 import BeautifulSoup, NavigableString, Comment

        soup = BeautifulSoup(html, 'html.parser')
        for element in soup(['script', 'style', 'head']):
            element.decompose()

        # Preserve anchor links as <a href="...">text</a> instead of flattening
        # to bare text. Transactional emails (booking confirmations, etc.) put
        # cancel/manage actions behind <a> buttons; get_text() would drop the
        # href and leave the assistant with a label and no URL to follow.
        def extract(element) -> str:
            parts = []
            for child in element.children:
                if isinstance(child, Comment):
                    continue
                if isinstance(child, NavigableString):
                    text = str(child).strip()
                    if text:
                        parts.append(text)
                elif child.name == 'a' and child.get('href'):
                    href = child['href']
                    link_text = child.get_text(strip=True)
                    if href.startswith(('http://', 'https://', 'mailto:')):
                        label = link_text or href
                        parts.append(f'<a href="{href}">{label}</a>')
                    elif link_text:
                        parts.append(link_text)
                elif child.name == 'br':
                    parts.append('\n')
                elif child.name in ('p', 'div', 'tr', 'li', 'table', 'section',
                                    'h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                    inner = extract(child)
                    if inner:
                        parts.append(inner)
                        parts.append('\n')
                else:
                    inner = extract(child)
                    if inner:
                        parts.append(inner)
            return ' '.join(parts)

        root = soup.find('body') or soup
        text = extract(root)
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        return '\n'.join(lines)

    @staticmethod
    def strip_html_plain(html: str) -> str:
        """Flatten HTML to visible text only, dropping all href URLs.

        Use this when the consumer needs the human-readable text without link
        markup — e.g. regex-scanning for a verification code, where an embedded
        URL containing digits would produce false matches. For email bodies
        shown to the assistant, prefer link-preserving strip_html().
        """
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        for element in soup(['script', 'style', 'head']):
            element.decompose()
        text = soup.get_text('\n', strip=True)
        lines = [line for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)

    @staticmethod
    def _extract_links(html: str) -> List[str]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, 'html.parser')
        for element in soup(['script', 'style', 'head']):
            element.decompose()
        urls = []
        seen = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith(('http://', 'https://')) and href not in seen:
                seen.add(href)
                urls.append(href)
        return urls

    def _format_email_body(self, msg) -> str:
        if msg.text:
            body = msg.text
            # The text/plain part often omits the deep links (cancel/manage
            # booking, etc.) that live only in the HTML alternative. Surface any
            # HTML links not already present in the chosen body.
            if msg.html:
                missing = [u for u in self._extract_links(msg.html) if u not in body]
                if missing:
                    body += "\n\nLinks:\n" + "\n".join(missing)
        elif msg.html:
            body = self.strip_html(msg.html)
        else:
            body = "(no body)"
        if len(body) > self.MAX_BODY_LENGTH:
            body = body[:self.MAX_BODY_LENGTH] + "\n... (truncated)"

        attachments_text = self._process_attachments(msg)

        result = (
            f"From: {msg.from_}\n"
            f"To: {', '.join(msg.to)}\n"
            f"Date: {self._format_date(msg.date)}\n"
            f"Subject: {msg.subject}\n"
            f"\n{body}"
        )
        if attachments_text:
            result += f"\n\n{attachments_text}"
        return result

    @staticmethod
    def _process_attachments(msg) -> str:
        if not msg.attachments:
            return ""

        sections = []
        for att in msg.attachments:
            content_type = att.content_type or ""
            filename = att.filename or "unnamed"
            size = att.size or 0
            size_kb = size / 1024

            kind = _attachment_kind(att)
            if kind == KIND_PDF:
                info(f"Email attachment: extracting PDF '{filename}' ({size_kb:.0f} KB, type '{content_type}')")
                text = PdfExtractor.extract_text(att.payload)
                sections.append(f"--- Attachment: {filename} (PDF, {size_kb:.0f} KB) ---\n{text}")

            elif kind is not None:
                if _is_junk_media(att):
                    trace(f"Email attachment: skipping small/inline/unsupported {kind} '{filename}' ({size_kb:.0f} KB)")
                    continue
                info(f"Email attachment: analyzing {kind} '{filename}' ({size_kb:.0f} KB, type '{content_type}')")
                description = _analyze_media(att.payload, content_type, filename)
                sections.append(f"--- Attachment: {filename} ({kind}, {size_kb:.0f} KB) ---\n{description}")

            else:
                info(f"Email attachment: unsupported type '{content_type}' for '{filename}' ({size_kb:.0f} KB), "
                     f"telling the reader it was not read")
                sections.append(f"[attachment {filename} not read: type {content_type or 'unknown'}, {size_kb:.0f} KB]")

        return "\n\n".join(sections)
