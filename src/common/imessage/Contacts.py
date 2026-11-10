import re
import sqlite3
import time
from pathlib import Path
from typing import Optional, Tuple

from common.imessage.Handles import is_group_guid, looks_like_handle, normalize_phone, phone_suffix, prefer_handle
from common.imessage.MessagesDatabase import MessagesDatabase
from common.imessage.MessagesHost import GroupChat, Recipient
from common.logging import info, trace, fine

ADDRESS_BOOK_DIR = Path.home() / "Library" / "Application Support" / "AddressBook"
CONTACT_CACHE_REFRESH_INTERVAL = 30 * 60
# how a recipient lists the members of an unnamed group: "Alice, Bob", "Alice & Bob", "Alice and Bob"
GROUP_MEMBER_SEPARATOR = re.compile(r"\s*(?:,|&|\+|\band\b|\boch\b)\s*", re.IGNORECASE)
GROUP_HINT_LIMIT = 3            # near-miss groups named in the error for a member list no group has

# Which of a contact's handles a send picks, best first: one chat.db knows as an iMessage
# handle (phone or email), a mobile number (AddressBook label or a Swedish 07x number), any other
# phone (landline, unlabelled), an email Messages has never seen, a fax/pager. Inside a tier the
# handle with the most recent conversation wins.
TIER_IMESSAGE, TIER_MOBILE, TIER_OTHER_PHONE, TIER_UNKNOWN_EMAIL, TIER_FAX = range(5)
TIER_NAMES = ("iMessage handle", "mobile", "other phone", "email unknown to Messages", "fax")
MOBILE_LABEL_PATTERN = re.compile(r"iphone|mobil|cell", re.IGNORECASE)     # "Mobile", "mobil" (sv)
FAX_LABEL_PATTERN = re.compile(r"fax|pager", re.IGNORECASE)
SWEDISH_MOBILE_PATTERN = re.compile(r"^(?:\+46|0046)7\d{8}$|^07\d{8}$")


class Contacts:
    """The Mac's AddressBook as Messages needs it: who a handle is, and which handle a name means.

    Reads the AddressBook's SQLite databases into a cache (refreshed when older than half an
    hour) that answers both ways: a phone or email to the contact's name, for the rows of
    chat.db, and a recipient — a contact name, a raw handle, a group's name or the list of its
    members — to the one handle a send addresses. A contact with several handles gets the one
    Messages has talked to most recently, read from the `MessagesDatabase`."""

    def __init__(self, database: MessagesDatabase, address_book: Optional[Path] = None):
        self._database = database
        self._address_book = address_book       # the AddressBook folder; None: the user's own
        self._contact_cache: dict[str, str] = {}
        self._name_to_handle: dict[str, str] = {}
        self._handle_reasons: dict[str, str] = {}     # name key → why its handle won (for the log)
        self._handle_to_full_name: dict[str, str] = {}
        self._card_names: dict[str, str] = {}       # handle key → the full name on its card, "" when cards disagree
        self._people_named: dict[str, int] = {}     # full name → how many different people carry it
        self._contact_cache_time = 0.0

    def __len__(self) -> int:
        return len(self._contact_cache)

    def refresh_if_stale(self):
        if time.time() - self._contact_cache_time > CONTACT_CACHE_REFRESH_INTERVAL:
            self.refresh()

    def name_for(self, handle_id: Optional[str]) -> str:
        """Resolve a handle (phone/email) to a contact name; the handle itself when no contact has it."""
        if not handle_id:
            return "Unknown"

        normalized = normalize_phone(handle_id)
        if normalized in self._contact_cache:
            return self._contact_cache[normalized]

        if handle_id in self._contact_cache:
            return self._contact_cache[handle_id]

        suffix = phone_suffix(handle_id)
        if suffix and suffix in self._contact_cache:
            return self._contact_cache[suffix]

        return handle_id

    def full_name(self, handle: str) -> Optional[str]:
        """Canonical 'First Last' for a resolved handle, or None if unknown."""
        if not handle:
            return None
        return self._handle_to_full_name.get(handle)

    @staticmethod
    def _handle_keys(handle: str) -> list[str]:
        """What a handle is filed under so it is found in whatever format it is written: an
        email in lower case; a phone by its digits and by its last nine."""
        if "@" in handle:
            return [handle.strip().lower()]
        return [key for key in (normalize_phone(handle), phone_suffix(handle)) if key]

    @classmethod
    def _note_card(cls, cards: dict, card: tuple, first, middle, last, handle: str):
        """Add a handle to its card: the card's full name and every key of its handles."""
        full = " ".join(part.strip() for part in (first, middle, last) if part and part.strip())
        if full and handle:
            cards.setdefault(card, (full, set()))[1].update(cls._handle_keys(handle))

    @staticmethod
    def _names_by_handle(cards: dict) -> dict[str, str]:
        """Handle key → the full name of the card it is on; "" for a handle on cards with
        different names (a landline two people share), which names nobody for certain."""
        names: dict[str, set[str]] = {}
        for full, keys in cards.values():
            for key in keys:
                names.setdefault(key, set()).add(full)
        return {key: next(iter(fulls)) if len(fulls) == 1 else "" for key, fulls in names.items()}

    @staticmethod
    def _count_people(cards: dict) -> dict[str, int]:
        """How many different people carry each full name. Two cards with one name are one
        person when they share a handle — the same contact in two accounts — and two when
        they do not."""
        people: dict[str, list[set[str]]] = {}
        for full, keys in cards.values():
            merged = people.setdefault(full, [])
            for other in [known for known in merged if known & keys]:
                merged.remove(other)
                keys = keys | other
            merged.append(keys)
        return {full: len(merged) for full, merged in people.items()}

    def sidebar_name(self, chat: str) -> tuple[str, str]:
        """(what the Messages sidebar of this Mac calls a chat — `chat.guid`, or a bare handle
        —, "") or ("", why this Mac cannot tell or the name is not that chat's alone). A named
        group goes by its name, a person by the full name of their card, a handle no card has
        by the handle. The typing dots find a chat's window by this name, so a name two chats
        could answer to is no name: another group called the same, a contact called like a
        group, two different people with one name."""
        self.refresh_if_stale()
        groups = self.group_chats()
        if is_group_guid(chat):
            group = next((g for g in groups if g.guid == chat), None)
            if group is None:
                return "", "chat.db has no such group chat"
            if not group.name:
                return "", "the group has no name — what the sidebar calls an unnamed group is not known"
            twins = sum(1 for g in groups if g.name == group.name)
            if twins > 1:
                return "", f"{twins} group chats of this account are called '{group.name}'"
            if self._people_named.get(group.name):
                return "", f"a contact is called '{group.name}' as well as the group"
            return group.name, ""
        handle = (chat or "").rsplit(";", 1)[-1].strip()
        if not handle:
            return "", "the chat names nobody"
        full = next((self._card_names[key] for key in self._handle_keys(handle) if key in self._card_names), None)
        if full is None:
            if self.name_for(handle) != handle:
                return "", "the handle is a contact's whose full name is not known"
            return prefer_handle(handle) or handle, ""
        if not full:
            return "", "the handle is on the cards of contacts with different names"
        if self._people_named.get(full, 0) > 1:
            return "", f"{self._people_named[full]} different contacts are called '{full}'"
        if any(g.name == full for g in groups):
            return "", f"a group chat is called '{full}' as well as the contact"
        return full, ""

    @staticmethod
    def _add_candidate(candidates: dict[str, list[tuple[str, Optional[str]]]], name: str,
                       raw_handle: str, label: Optional[str] = None):
        """Add a handle (with its AddressBook label) to the candidates of a name. The key is the
        lowercased, whitespace-collapsed name — callers pass first name, full name, etc. as
        separate calls; `_rank_candidates` picks one handle per name once the cards are read."""
        if not name or not raw_handle:
            return
        key = " ".join(name.split()).lower()
        handle = prefer_handle(raw_handle)
        if not key or not handle:
            return
        entries = candidates.setdefault(key, [])
        if all(h != handle for h, _ in entries):
            entries.append((handle, label))

    @staticmethod
    def _record_full_name(mapping: dict[str, str], first: str | None, last: str | None, raw_handle: str,
                          middle: str | None = None):
        """Map handle → the card's full name ('First Middle Last', the parts that are set). Used to
        render canonical names in approval dialogs even when the caller passed only a first name."""
        if not first or not raw_handle:
            return
        handle = prefer_handle(raw_handle)
        if not handle:
            return
        parts = [part.strip() for part in (first, middle, last) if part and part.strip()]
        if not parts:
            return
        mapping.setdefault(handle, " ".join(parts))

    @staticmethod
    def _name_keys(first: str | None, last: str | None, middle: str | None = None) -> list[str]:
        """Lookup keys to register for a contact: the first name, then 'first last', and with a middle
        name on the card also 'first middle' and 'first middle last' — a Swedish double surname is
        often filed as middle + last, and the user says either half. Last-name-only is intentionally
        omitted — too easy to collide across contacts."""
        f = (first or "").strip()
        m = (middle or "").strip()
        l = (last or "").strip()
        if not f:
            return []
        keys = [f]
        if m:
            keys.append(f"{f} {m}")
        if l:
            keys.append(f"{f} {l}")
        if m and l:
            keys.append(f"{f} {m} {l}")
        return keys

    @staticmethod
    def _handle_priority(handle: str) -> int:
        """Lower is better. Prefer E.164 phones, then emails, then bare digits."""
        if not handle:
            return 99
        if handle.startswith("+"):
            return 0
        if "@" in handle:
            return 1
        return 2

    @staticmethod
    def _handle_tier(handle: str, label: Optional[str], activity: Optional[tuple[bool, int, int]]) -> int:
        """The preference tier of a handle (`TIER_*`, lower is better) given its AddressBook label
        and what chat.db knows about it."""
        if activity and activity[0]:
            return TIER_IMESSAGE
        if "@" in handle:
            return TIER_UNKNOWN_EMAIL
        if label and FAX_LABEL_PATTERN.search(label):
            return TIER_FAX
        if (label and MOBILE_LABEL_PATTERN.search(label)) or SWEDISH_MOBILE_PATTERN.match(handle):
            return TIER_MOBILE
        return TIER_OTHER_PHONE

    @staticmethod
    def _lookup_activity(activity: dict[str, tuple[bool, int, int]], handle: str) \
            -> Optional[tuple[bool, int, int]]:
        """The activity of a handle. A phone is looked up by its suffix first: that entry merges every
        chat.db row of the number whatever its format, so a local-format id Messages once stored
        verbatim does not hide the history of its E.164 twin."""
        suffix = phone_suffix(handle) if "@" not in handle else None
        found = activity.get(suffix) if suffix else None
        return found if found is not None else activity.get(handle.lower())

    def _rank_candidates(self, candidates: dict[str, list[tuple[str, Optional[str]]]]) \
            -> tuple[dict[str, str], dict[str, str]]:
        """Pick one handle per name: the best tier, then the most recent conversation in chat.db
        (then the most messages), then E.164 over an email over bare digits, then the card's
        order. Returns the name → handle map and, per name, why the handle won — the line
        `resolve` logs. chat.db is read only for the handles of the names with a choice
        to make, so the fallback lookup of one name costs one small query; when it cannot be
        read the ranking falls back to the card alone."""
        reverse: dict[str, str] = {}
        reasons: dict[str, str] = {}
        contested = {key: entries for key, entries in candidates.items() if len(entries) > 1}
        contested_handles = {handle for entries in contested.values() for handle, _ in entries}
        activity = self._database.chat_activity(contested_handles) if contested else {}
        for key, entries in candidates.items():
            ranked = []
            for index, (handle, label) in enumerate(entries):
                seen = self._lookup_activity(activity, handle)
                tier = self._handle_tier(handle, label, seen)
                last_date, count = (seen[1], seen[2]) if seen else (0, 0)
                ranked.append(((tier, -last_date, -count, self._handle_priority(handle), index), handle))
            (tier, neg_date, neg_count, _, index), handle = min(ranked)
            reverse[key] = handle
            history = f", {-neg_count} messages" if neg_count else ", no chat history"
            reasons[key] = f"{TIER_NAMES[tier]}{history}, listed {index + 1}/{len(entries)} on the card"
        if contested:
            overridden = [f"{key}→{reverse[key]} ({reasons[key]})" for key, entries in contested.items()
                          if reverse[key] != entries[0][0]]
            shown = "; ".join(overridden[:20]) + (" …" if len(overridden) > 20 else "")
            fine(f"iMessage: {len(contested)} contacts with several handles, {len(overridden)} resolved to "
                 f"one other than the card's first" + (f": {shown}" if overridden else ""))
        return reverse, reasons

    def _recipient(self, handle: str) -> Recipient:
        name = self.name_for(handle)
        return Recipient(handle=handle, name=name if name != handle else None, full_name=self.full_name(handle))

    def resolve(self, recipient: str, owner: str = "") -> Recipient:
        """Resolve a recipient string (contact name, raw handle, group name or member list) to a
        sendable handle — for a group chat its guid, see `Handles.is_group_guid`. On failure the
        handle is None and the error a short human-readable string."""
        if not recipient or not recipient.strip():
            return Recipient(error="empty recipient")
        candidate = recipient.strip()

        if looks_like_handle(candidate):
            return self._recipient(prefer_handle(candidate) or candidate)

        self.refresh_if_stale()

        key = " ".join(candidate.split()).lower()
        handle = self._name_to_handle.get(key)
        if handle:
            info(f"iMessage: resolved '{candidate}' → {handle} ({self._handle_reasons.get(key, 'only handle')})")
            return self._recipient(handle)

        fallback = self._fallback_lookup(key)
        if fallback:
            info(f"iMessage: resolved '{candidate}' via forward-cache fallback → {fallback}")
            return self._recipient(fallback)

        group, group_err = self._resolve_group(candidate, owner)
        if group:
            return Recipient(handle=group.guid, group=group)
        if group_err:
            return Recipient(error=group_err)

        if not self._contact_cache and not self._name_to_handle:
            info(f"iMessage: contact cache is empty, '{candidate}' cannot be looked up")
            return Recipient(error=f"contact cache is empty — could not look up '{candidate}'", no_contacts=True)
        return Recipient(error=f"unknown contact: {candidate} (cache has {len(self._name_to_handle)} names)")

    def group_chats(self) -> list[GroupChat]:
        """Every group chat with its members named like the feed names them."""
        return self._database.group_chats(self.name_for)

    @staticmethod
    def _member_matches(member: str, part: str) -> bool:
        """A listed member matches a contact by full name or first name ("Alice" → "Alice Smith")."""
        name = member.lower()
        return name == part or name.split()[0] == part

    @classmethod
    def _has_members(cls, group: GroupChat, wanted: set[str]) -> bool:
        return all(any(cls._member_matches(m, p) for m in group.members) for p in wanted)

    @classmethod
    def _has_exactly_members(cls, group: GroupChat, wanted: set[str]) -> bool:
        return cls._has_members(group, wanted) and \
            all(any(cls._member_matches(m, p) for p in wanted) for m in group.members)

    def _resolve_group(self, recipient: str, owner: str = "") -> Tuple[Optional[GroupChat], Optional[str]]:
        """The group chat a recipient names — by its display name, or for an unnamed group by its
        members the way the feed lists them ("Alice, Bob", "Alice & Bob", "Alice and Bob"; the
        owner's own name may be in the list). Several chats with the same name or members (an
        iMessage and an SMS incarnation, a recreated group): the most recently active one wins.
        Returns (group, error): an error only for a member list no group has exactly, naming the
        groups that contain those people; (None, None) when the recipient is not a group's name."""
        groups = self.group_chats()
        if not groups:
            return None, None
        key = " ".join(recipient.split()).lower()
        wanted = {part.strip().lower() for part in GROUP_MEMBER_SEPARATOR.split(recipient)} \
            - {"", (owner or "").strip().lower()}
        matches = [g for g in groups if g.name and g.name.lower() == key]
        how = "by name"
        if not matches and len(wanted) >= 2:
            matches = [g for g in groups if self._has_exactly_members(g, wanted)]
            how = "by members"
        if matches:
            group = max(matches, key=lambda g: g.last_active)
            info(f"iMessage: resolved '{recipient}' {how} → group {group.describe()} ({group.guid}, "
                 f"{len(matches)} candidate(s))")
            return group, None
        if len(wanted) < 2:
            trace(f"iMessage: '{recipient}' names no group chat")
            return None, None
        near = sorted((g for g in groups if self._has_members(g, wanted)),
                      key=lambda g: g.last_active, reverse=True)[:GROUP_HINT_LIMIT]
        info(f"iMessage: no group chat with exactly {sorted(wanted)}; {len(near)} group(s) contain them")
        hint = f"; groups with them: {'; '.join(g.describe() for g in near)}" if near else ""
        return None, f"no group chat with exactly these members: {', '.join(sorted(wanted))}{hint}"

    def _fallback_lookup(self, name_lower: str) -> Optional[str]:
        """Last-resort: scan the forward (handle → name) cache for any handle whose name matches
        and rank them like the card's handles (without labels — the forward cache has none).
        Useful when the reverse cache missed for any reason."""
        if not self._contact_cache:
            return None
        candidates: dict[str, list[tuple[str, Optional[str]]]] = {}
        for raw_handle, contact_name in self._contact_cache.items():
            if contact_name and contact_name.strip().lower() == name_lower:
                self._add_candidate(candidates, name_lower, raw_handle)
        if not candidates:
            return None
        reverse, reasons = self._rank_candidates(candidates)
        trace(f"iMessage: fallback lookup of '{name_lower}' → {reverse[name_lower]} ({reasons[name_lower]})")
        return reverse[name_lower]

    def refresh(self):
        """Build phone/email -> name cache by reading the macOS AddressBook SQLite databases directly."""
        try:
            ab_root = self._address_book or ADDRESS_BOOK_DIR
            source_dbs = list((ab_root / "Sources").glob("*/AddressBook-v22.abcddb"))
            if not source_dbs:
                trace("iMessage: no AddressBook source databases found")
                self._contact_cache_time = time.time()
                return

            cache: dict[str, str] = {}
            candidates: dict[str, list[tuple[str, Optional[str]]]] = {}
            handle_to_full: dict[str, str] = {}
            cards: dict[tuple, tuple[str, set[str]]] = {}   # (source, card) → (full name, the keys of its handles)
            for db_path in source_dbs:
                try:
                    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                        for card, first, middle, last, phone, label in conn.execute("""
                            SELECT r.Z_PK, r.ZFIRSTNAME, r.ZMIDDLENAME, r.ZLASTNAME, p.ZFULLNUMBER, p.ZLABEL
                            FROM ZABCDRECORD r JOIN ZABCDPHONENUMBER p ON p.ZOWNER = r.Z_PK
                            WHERE r.ZFIRSTNAME IS NOT NULL AND p.ZFULLNUMBER IS NOT NULL
                            ORDER BY r.Z_PK, p.Z_PK
                        """):
                            self._note_card(cards, (db_path, card), first, middle, last, phone)
                            normalized = normalize_phone(phone)
                            if normalized:
                                cache[normalized] = first
                            suffix = phone_suffix(phone)
                            if suffix:
                                cache[suffix] = first
                            for name_key in self._name_keys(first, last, middle):
                                self._add_candidate(candidates, name_key, phone, label)
                            self._record_full_name(handle_to_full, first, last, phone, middle)

                        for card, first, middle, last, email in conn.execute("""
                            SELECT r.Z_PK, r.ZFIRSTNAME, r.ZMIDDLENAME, r.ZLASTNAME, e.ZADDRESS
                            FROM ZABCDRECORD r JOIN ZABCDEMAILADDRESS e ON e.ZOWNER = r.Z_PK
                            WHERE r.ZFIRSTNAME IS NOT NULL AND e.ZADDRESS IS NOT NULL
                        """):
                            self._note_card(cards, (db_path, card), first, middle, last, email)
                            cache[email.lower()] = first
                            for name_key in self._name_keys(first, last, middle):
                                self._add_candidate(candidates, name_key, email)
                            self._record_full_name(handle_to_full, first, last, email, middle)
                except Exception as e:
                    fine(f"iMessage: skipping contact source {db_path.parent.name}: {e}")

            reverse, reasons = self._rank_candidates(candidates)
            self._contact_cache = cache
            self._name_to_handle = reverse
            self._handle_reasons = reasons
            self._handle_to_full_name = handle_to_full
            self._card_names = self._names_by_handle(cards)
            self._people_named = self._count_people(cards)
            self._contact_cache_time = time.time()
            fine(f"iMessage: refreshed contact cache ({len(cache)} entries, {len(reverse)} reverse)")
        except Exception as e:
            trace(f"iMessage: failed to refresh contact cache: {e}")
            self._contact_cache_time = time.time()
