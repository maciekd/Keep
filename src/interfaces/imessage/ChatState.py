"""What the iMessage chat transport must still know after a restart, kept in one small file under `cache/`.

The cursor — the chat.db ROWID the poll has read up to, with the account it was read from: a
ROWID of another Mac's chat.db means nothing. The groups in which the assistant wrote last,
so the first answer to it after a restart still wakes it. The groups an owner has written in,
which is what makes a group a place the assistant may post to, and the names those groups were
posted to under: a name that was one group's is never handed to another. And how many messages
went to each recipient today, so a restart does not hand a runaway loop a fresh daily allowance.
"""
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from common.Paths import CACHE_DIR
from common.logging import error, info, trace

STATE_PATH = CACHE_DIR / "imessage_chat.json"
NAMES_PER_GROUP = 8     # the names kept for one group, latest last: a group renamed for ever does not grow the file


def today(now: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(now))


class ChatState:
    """What the iMessage chat transport keeps across a restart: its cursor, the groups it wrote last in, the groups an owner vouched for with their names, and the day's send counts.

    The state of one account as one JSON object, written whole after every change. A file
    that is about another account — `imessage_chat.host` was pointed elsewhere — is started over."""

    def __init__(self, account: str, path: Optional[Path] = None):
        self.account = account
        self.path = Path(path) if path is not None else STATE_PATH
        self._lock = threading.Lock()
        self._data = self._read()
        if self._data.get("account") != account:
            if self._data:
                info(f"iMessage chat: the state at {self.path} is about another account — starting over for '{account}'")
            self._data = {"account": account}

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            trace(f"iMessage chat: no state at {self.path} yet")
            return {}
        except (OSError, ValueError) as e:
            error(f"iMessage chat: the state at {self.path} cannot be read ({e}) — starting without one")
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            scratch = self.path.with_suffix(".tmp")
            scratch.write_text(json.dumps(self._data), encoding="utf-8")
            os.replace(scratch, self.path)
        except OSError as e:
            error(f"iMessage chat: the state could not be written to {self.path}: {e}")

    def cursor(self) -> Optional[int]:
        """The chat.db ROWID the poll stopped at, None when it never ran on this account."""
        with self._lock:
            cursor = self._data.get("cursor")
            return cursor if isinstance(cursor, int) and not isinstance(cursor, bool) else None

    def save_cursor(self, cursor: int):
        with self._lock:
            if self._data.get("cursor") == cursor:
                return
            self._data["cursor"] = cursor
            self._write()

    def following(self) -> list[str]:
        """The groups whose latest message was the assistant's own when the state was written."""
        with self._lock:
            return [g for g in self._data.get("following", []) if isinstance(g, str)]

    def save_following(self, groups: list[str]):
        groups = sorted(groups)
        with self._lock:
            if self._data.get("following", []) == groups:
                return
            self._data["following"] = groups
            self._write()

    def vouched(self) -> dict[str, list[str]]:
        """The groups an owner has written in over iMessage, each with the handles that did."""
        with self._lock:
            groups = self._data.get("vouched", {})
            if not isinstance(groups, dict):
                return {}
            return {guid: [h for h in handles if isinstance(h, str)]
                    for guid, handles in groups.items() if isinstance(guid, str) and isinstance(handles, list)}

    def vouch(self, group: str, handle: str) -> bool:
        """An owner writing from *handle* wrote in *group*; True when that is new."""
        with self._lock:
            groups = self._data.get("vouched")
            if not isinstance(groups, dict):
                groups = self._data["vouched"] = {}
            handles = groups.setdefault(group, [])
            if handle in handles:
                return False
            handles.append(handle)
            self._write()
            return True

    def unvouch(self, group: str, handles: Optional[list[str]] = None) -> bool:
        """Forget that *handles* wrote in *group* — all who did when left out; True when
        something was forgotten. A group nobody vouches for keeps no entry."""
        with self._lock:
            groups = self._data.get("vouched")
            known = groups.get(group) if isinstance(groups, dict) else None
            if not isinstance(known, list):
                return False
            left = [] if handles is None else [h for h in known if h not in handles]
            if len(left) == len(known):
                return False
            if left:
                groups[group] = left
            else:
                del groups[group]
            self._write()
            return True

    def names(self) -> dict[str, str]:
        """The destination names groups were registered under, each with the guid of its group."""
        with self._lock:
            names = self._data.get("names", {})
            if not isinstance(names, dict):
                return {}
            return {name: guid for name, guid in names.items() if isinstance(name, str) and isinstance(guid, str)}

    def keep_name(self, name: str, group: str):
        """*name* (as destinations compare names) is *group*'s from now on."""
        with self._lock:
            names = self._data.get("names")
            if not isinstance(names, dict):
                names = self._data["names"] = {}
            if names.get(name) == group:
                return
            names.pop(name, None)
            names[name] = group
            own = [known for known, guid in names.items() if guid == group]
            for old in own[:-NAMES_PER_GROUP]:
                del names[old]
            self._write()

    def free_names(self, groups) -> list[str]:
        """Give up the names of every group but *groups*; the names given up."""
        with self._lock:
            names = self._data.get("names")
            if not isinstance(names, dict):
                return []
            freed = [name for name, guid in names.items() if guid not in groups]
            for name in freed:
                del names[name]
            if freed:
                self._write()
            return freed

    def sent_today(self, recipient: str, now: Optional[float] = None) -> int:
        with self._lock:
            sent = self._data.get("sent", {})
            return int(sent.get("counts", {}).get(recipient, 0)) if sent.get("day") == today(now) else 0

    def count_sent(self, recipient: str, now: Optional[float] = None) -> int:
        """One more message to *recipient* today; the count after it."""
        with self._lock:
            sent = self._data.get("sent", {})
            if sent.get("day") != today(now):
                sent = {"day": today(now), "counts": {}}
            sent["counts"][recipient] = int(sent["counts"].get(recipient, 0)) + 1
            self._data["sent"] = sent
            self._write()
            return sent["counts"][recipient]
