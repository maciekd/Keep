"""One-off rewrites of config.yaml itself, run by ConfigManager before the sections are read.

2026-10-07: the mail and calendar integrations hold accounts. The `email` (Gmail), `icloud_email`
and `calendar` (iCloud CalDAV) sections became entries of `imap.accounts` and `caldav.accounts`,
and `Gmail` / `iCloud Email` in `enabled_integrations` became `IMAP Mail`. An old section is
moved, then left in place commented out (`# migrated <date>`), so the file explains itself and a
second run finds nothing to do. ruamel keeps every other comment. What moved is logged line by
line and summarised in `pending_note`, which the launcher posts as a system note once the dialog
history is up (this module is `common`, below the assistant)."""
import io
import re
import time
from pathlib import Path
from typing import Any, Optional

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from common import PrivateFile
from common.Paths import CACHE_DIR
from common.logging import info, trace, error

ENABLED_KEY = "enabled_integrations"
IMAP_INTEGRATION = "IMAP Mail"
OLD_MAIL_INTEGRATIONS = ("gmail", "icloudemail")
# old section -> the name and preset of the imap account it becomes
MAIL_SECTIONS = {"email": ("Gmail", "gmail"), "icloud_email": ("iCloud", "icloud")}
CALENDAR_SECTION = "calendar"
BACKUP_DIR = CACHE_DIR / "config-backups"

# The note for the household, set by the last run that changed something and taken once.
pending_note: Optional[str] = None


def _integration_key(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", re.sub(r"&|\band\b", "", str(name).lower()))


def needed(data: dict) -> bool:
    """Whether a loaded config.yaml (plain dicts) still has something to migrate."""
    if not isinstance(data, dict):
        return False
    if any(isinstance(data.get(section), dict) for section in (*MAIL_SECTIONS, CALENDAR_SECTION)):
        return True
    listed = data.get(ENABLED_KEY) or []
    listed = [listed] if isinstance(listed, str) else listed
    return isinstance(listed, list) and any(_integration_key(n) in OLD_MAIL_INTEGRATIONS for n in listed)


def _yaml() -> YAML:
    y = YAML(typ="rt")
    y.preserve_quotes = True
    y.width = 4096
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def _commented_out(yaml: YAML, section: str, node, date_text: str) -> str:
    """The old section as YAML text, every line a comment, headed by the migration date."""
    buffer = io.StringIO()
    yaml.dump(CommentedMap({section: node}), buffer)
    # ruamel prefixes each line with "# " itself when the text is set as a comment.
    lines = [f"migrated {date_text}: `{section}` became an account entry below, kept for reference"]
    lines += buffer.getvalue().rstrip("\n").splitlines()
    return "\n".join(lines)


def _accounts_list(data: CommentedMap, section: str, position: int) -> CommentedSeq:
    parent = data.get(section)
    if not isinstance(parent, dict):
        parent = CommentedMap()
        data.insert(position, section, parent)
    accounts = parent.get("accounts")
    if not isinstance(accounts, list):
        accounts = parent["accounts"] = CommentedSeq()
    return accounts


# Lives here, not in ConfigStore: the migration runs while ConfigManager's module-level instance
# loads, which can be in the middle of ConfigStore's own import (the launcher imports Migrations →
# ConfigStore → ConfigManager), so nothing above ConfigManager may be imported from here.
_PLACEHOLDER_RE = re.compile(r"YOUR_|CHANGE_ME|_HERE\b|DEVICE_UUID", re.IGNORECASE)


def looks_like_placeholder(value: Any) -> bool:
    """A value still holding the example's placeholder (YOUR_APP_PASSWORD, CHANGE_ME, DEVICE_UUID…),
    looked for inside lists and mappings too."""
    if isinstance(value, str):
        return bool(_PLACEHOLDER_RE.search(value))
    if isinstance(value, dict):
        return any(looks_like_placeholder(v) for v in value.values())
    if isinstance(value, list):
        return any(looks_like_placeholder(v) for v in value)
    return False


def _account(fields: list[tuple[str, object]]) -> CommentedMap:
    """The entry out of an old section's fields: an empty value or a copied example placeholder
    (YOUR_APP_PASSWORD) is left out, so a never-filled section carries nothing over."""
    account = CommentedMap()
    for key, value in fields:
        if value not in (None, "", []) and not looks_like_placeholder(value):
            account[key] = value
    return account


def migrate(data: CommentedMap, date_text: str) -> list[str]:
    """Apply every mapping to the round-trip tree in place; returns one line per change."""
    done: list[str] = []
    yaml = _yaml()
    comments: dict[str, list[str]] = {}
    migrated_icloud_mail = False
    for section, (name, preset) in MAIL_SECTIONS.items():
        old = data.get(section)
        if not isinstance(old, dict):
            continue
        position = list(data.keys()).index(section)
        comments.setdefault("imap", []).append(_commented_out(yaml, section, old, date_text))
        del data[section]
        data.ca.items.pop(section, None)
        accounts = _accounts_list(data, "imap", position)
        account = _account([("name", name), ("preset", preset), ("address", old.get("address")),
                            ("password", old.get("password")), ("smtp_host", old.get("smtp_host")),
                            ("smtp_port", old.get("smtp_port"))])
        if "address" not in account:
            done.append(f"{section} held no address (empty or the example's placeholders) — no account made of it")
            continue
        accounts.append(account)
        done.append(f"{section} → imap.accounts[{len(accounts) - 1}] (name {name}, preset {preset})")
        migrated_icloud_mail |= preset == "icloud"
    old = data.get(CALENDAR_SECTION)
    if isinstance(old, dict):
        position = list(data.keys()).index(CALENDAR_SECTION)
        comments.setdefault("caldav", []).append(_commented_out(yaml, CALENDAR_SECTION, old, date_text))
        del data[CALENDAR_SECTION]
        data.ca.items.pop(CALENDAR_SECTION, None)
        accounts = _accounts_list(data, "caldav", position)
        account = _account([("name", "iCloud"), ("preset", "icloud"), ("username", old.get("username")),
                            ("password", old.get("password")),
                            ("default_calendar", old.get("default_calendar") or "Home"),
                            ("shown_calendars", old.get("shown_calendars", []))])
        if "username" not in account:
            done.append(f"{CALENDAR_SECTION} held no username (empty or the example's placeholders) — no account made of it")
        else:
            accounts.append(account)
            done.append(f"{CALENDAR_SECTION} → caldav.accounts[{len(accounts) - 1}] (name iCloud, preset icloud)")
    for section, texts in comments.items():
        data.yaml_set_comment_before_after_key(section, before="\n".join(texts))

    listed = data.get(ENABLED_KEY)
    if isinstance(listed, str):
        listed = data[ENABLED_KEY] = CommentedSeq([listed])
    if isinstance(listed, list):
        # The first old name takes IMAP Mail's place in the list, the others go.
        has_imap = any(_integration_key(n) == _integration_key(IMAP_INTEGRATION) for n in listed)
        surplus = []
        for i, old_name in enumerate(listed):
            if _integration_key(old_name) not in OLD_MAIL_INTEGRATIONS:
                continue
            if has_imap:
                surplus.append(i)
                done.append(f"{ENABLED_KEY}: {old_name} removed ({IMAP_INTEGRATION} is listed)")
            else:
                listed[i] = IMAP_INTEGRATION
                has_imap = True
                done.append(f"{ENABLED_KEY}: {old_name} → {IMAP_INTEGRATION}")
        for i in reversed(surplus):
            del listed[i]
    if migrated_icloud_mail:
        mail = data.get("mail")
        if not isinstance(mail, dict):
            mail = CommentedMap()
            data.insert(list(data.keys()).index("imap") + 1, "mail", mail)
        if not mail.get("send_from"):
            mail["send_from"] = "iCloud"
            done.append("mail.send_from: iCloud (send_email sent from iCloud before)")
    return done


def _backup(path: Path) -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    target = BACKUP_DIR / f"config-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-before-migration.yaml"
    PrivateFile.copy(path, target)
    trace(f"Config migration: backed up to {target}")


def run(path: str) -> list[str]:
    """Migrate config.yaml at *path* when it still has the old sections; the file is rewritten
    only when something moved. Returns the change lines (empty when there was nothing to do)."""
    global pending_note
    file = Path(path)
    yaml = _yaml()
    try:
        data = yaml.load(file.read_text())
    except Exception as e:
        error(f"Config migration: {file.name} could not be read round-trip ({type(e).__name__}: {e}) — not migrated")
        return []
    if not isinstance(data, CommentedMap) or not needed(data):
        trace("Config migration: nothing to migrate")
        return []
    date_text = time.strftime("%Y-%m-%d")
    done = migrate(data, date_text)
    if not done:
        trace("Config migration: the old sections held nothing to move")
        return []
    _backup(file)
    dumped = io.StringIO()
    yaml.dump(data, dumped)
    PrivateFile.write(file, dumped.getvalue())
    for line in done:
        info(f"Config migration: {line}")
    pending_note = ("Config: the mail and calendar settings were moved to the new accounts layout — "
                    + "; ".join(done) + ". The old sections stay in config.yaml commented out.")
    return done


def take_note() -> Optional[str]:
    """The summary of the last run that changed something, once; None otherwise."""
    global pending_note
    note, pending_note = pending_note, None
    return note
