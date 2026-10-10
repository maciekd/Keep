"""Writes one value into config.yaml and says which keys are set — without ever handing a
value on. The assistant's `config_request` feeds `set_value` from the masked prompt; its
`list_integrations` / `config_help` render `describe_key`. plans/SELF_SETUP_PLAN.md §3.2.

Values are logged by key only, compared inside this module and discarded: nothing here
returns, formats or logs what the user typed or what config.yaml holds.
"""
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Optional

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from common import ConfigSchema
from common.ConfigManager import ENABLED_INTEGRATIONS_KEY, ConfigManager, integration_key
from common.ConfigMigrations import looks_like_placeholder  # noqa: F401 — the store's callers import it from here
from common.Paths import CACHE_DIR
from common.logging import info, trace

# Keys the assistant may not touch: changing vault.key locks the existing vault out, and
# interfaces.http is always on (the launcher never reads its flag — every other interface
# registers routes on it). The other interfaces.* flags are the user's to flip from the chat;
# they take effect at the next server restart (config_set says so). The last four are run
# as commands — over ssh on the Indigo Mac, in a shell before a photo opens — so a value the
# LLM could write (a prompt injection reaching config_set) would run within the next poll.
DENIED_PREFIXES = ("vault.key", "interfaces.http",
                   "indigo.ssh_user", "indigo.prompt_file", "indigo.log_refiner",
                   "photos.screen_prepare_command")
BACKUP_DIR = CACHE_DIR / "config-backups"
BACKUPS_KEPT = 10

# One writer at a time: every write is read-modify-replace through the same .tmp file,
# so two at once lose one change and race on the rename (2026-09-08 log).
_WRITE_LOCK = threading.Lock()
_TRUE = ("true", "yes", "on", "1")
_FALSE = ("false", "no", "off", "0")


class ConfigStoreError(Exception):
    """The value cannot be stored as given; the message is fit to show the user."""


def is_denied(key: str) -> bool:
    """Compared on the key's names, indexes aside: vault.key[0] and indigo.ssh_user[0].x
    would change a denied key as surely as the plain spelling does."""
    try:
        names = [p for p in ConfigSchema.split_key(key) if isinstance(p, str)]
    except ValueError:
        return False
    return any(names[:len(denied)] == denied for denied in (p.split(".") for p in DENIED_PREFIXES))


def check_key(key: str) -> None:
    """Raises ConfigStoreError unless *key* names a scalar setting the assistant may set.
    A list's items are addressed by index: icloud.devices[0].id for a list of sections,
    enabled_integrations[0] for a list of plain values; the list itself is not a single value."""
    key = key.strip()
    if is_denied(key):
        raise ConfigStoreError(f"'{key}' cannot be changed from here — edit config.yaml by hand")
    if not ConfigSchema.is_known_key(key):
        raise ConfigStoreError(f"'{key}' is not a known setting — config_help lists the sections")
    parts = ConfigSchema.split_key(key)
    if isinstance(parts[-1], int) and ConfigSchema.is_scalar_list(ConfigSchema.join_key(parts[:-1])):
        return
    kind = ConfigSchema.leaf_type(key)
    if kind is list and ConfigSchema.is_scalar_list(key):
        raise ConfigStoreError(f"'{key}' is a list of values — add one as {key}[{_list_length(key)}]")
    if kind is list:
        raise ConfigStoreError(f"'{key}' is a list — address one item's field, e.g. {key}[0].<field>")
    if kind is dict:
        raise ConfigStoreError(f"'{key}' is not a single value — set it in config.yaml by hand")


def _list_length(key: str) -> int:
    """How many items config.yaml holds at a list-of-values key — the next free index. A
    single value written by hand counts as one item, since a write promotes it to a list."""
    try:
        value = _current_value(key)
    except (KeyError, ValueError):
        return 0
    if value is None:
        return 0
    return len(value) if isinstance(value, list) else 1


# What `interfaces.<name>.enabled` means when config.yaml does not say — the launcher decides
# before importing an interface's module (voice pulls in the audio stack), so it cannot read
# the class's `enabled_by_default`; http is always on and never read.
INTERFACE_ENABLED_BY_DEFAULT = {"voice": False}
_INTERFACE_FLAG_RE = re.compile(r"^interfaces\.([a-z]+)\.enabled$")


def interface_flag(key: str) -> Optional[str]:
    """The interface name when *key* is an `interfaces.<name>.enabled` flag, else None."""
    m = _INTERFACE_FLAG_RE.match(key.strip())
    return m.group(1) if m else None


def interface_enabled(name: str) -> bool:
    """The effective `interfaces.<name>.enabled` right now: the config's value, else the default."""
    return bool(ConfigManager.get_instance().get(f"interfaces.{name}.enabled",
                                                 INTERFACE_ENABLED_BY_DEFAULT.get(name, True)))


def coerce(key: str, raw: str) -> Any:
    """The typed text as the type the example uses at *key* (str when the example has none)."""
    text = raw.strip()
    if not text:
        raise ConfigStoreError("the value is empty")
    kind = ConfigSchema.leaf_type(key)
    try:
        if kind is bool:
            if text.lower() in _TRUE:
                return True
            if text.lower() in _FALSE:
                return False
            raise ValueError
        if kind is int:
            return int(text)
        if kind is float:
            return float(text)
    except ValueError:
        raise ConfigStoreError(f"'{key}' expects a {kind.__name__}")
    return text


def _yaml() -> YAML:
    y = YAML(typ="rt")
    y.preserve_quotes = True
    y.width = 4096
    y.indent(mapping=2, sequence=4, offset=2)   # round-trips config.yaml byte for byte (verified 2026-09-08)
    return y


def _stamp() -> str:
    # Microseconds: a config lane batch writes several keys within one second, and the
    # backup taken before the first write must survive the ones after it.
    now_ns = time.time_ns()
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime(now_ns // 1_000_000_000)) + f"-{now_ns // 1000 % 1_000_000:06d}"


def _backup(path: Path) -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    target = BACKUP_DIR / f"config-{_stamp()}.yaml"
    shutil.copy2(path, target)
    old = sorted(BACKUP_DIR.glob("config-*.yaml"))[:-BACKUPS_KEPT]
    for stale in old:
        stale.unlink()
    trace(f"Config: backed up to {target.name}" + (f", dropped {len(old)} old backup(s)" if old else ""))


def _reload() -> None:
    from common import ConfigManager as module
    module.config.reload()
    current = ConfigManager.get_instance()
    if current is not module.config:
        current.reload()


def _parent_node(data: CommentedMap, key: str):
    """The mapping that holds *key*'s last segment, creating the path on the way. A list
    index equal to the list's length appends a new item; a larger one is an error. A path
    segment that already holds something of another shape (a list where the key expects a
    mapping, a scalar where it expects either) is refused rather than replaced: the slip
    `icloud.devices.id` must not wipe the device list. An empty (null) node is filled in."""
    parts = ConfigSchema.split_key(key)
    if not isinstance(parts[-1], str) and not ConfigSchema.is_scalar_list(ConfigSchema.join_key(parts[:-1])):
        raise ConfigStoreError(f"'{key}' names a whole list item — set one of its fields")
    node = data
    for i, part in enumerate(parts[:-1]):
        where = ConfigSchema.join_key(parts[:i + 1])
        wants_list = isinstance(parts[i + 1], int)
        if isinstance(part, int):
            if part > len(node):
                raise ConfigStoreError(f"{ConfigSchema.join_key(parts[:i])} has {len(node)} item(s) — [{len(node)}] adds the next one")
            if part == len(node):
                node.append(CommentedMap())
            child = node[part]
            if child is None:
                child = node[part] = CommentedMap()
        else:
            child = node.get(part)
            if child is None:
                child = node[part] = CommentedSeq() if wants_list else CommentedMap()
            elif wants_list and isinstance(child, str) and ConfigSchema.is_scalar_list(where):
                child = node[part] = _promoted(child, where)
        if wants_list and not isinstance(child, list):
            raise ConfigStoreError(f"'{where}' is not a list in config.yaml — set it by hand")
        if not wants_list and not isinstance(child, dict):
            hint = f"address one item, e.g. {where}[0].{parts[i + 1]}" if isinstance(child, list) else "set it by hand"
            raise ConfigStoreError(f"'{where}' is not a section in config.yaml — {hint}")
        node = child
    return node


def _rewrite(mutate) -> None:
    """Load config.yaml round-trip, apply *mutate(data)*, back the file up, write it
    atomically and reload the running config. One writer at a time."""
    with _WRITE_LOCK:
        _rewrite_locked(mutate)


def _rewrite_locked(mutate) -> None:
    path = Path(ConfigManager.config_path())
    yaml = _yaml()
    data = yaml.load(path.read_text()) if path.exists() else None
    if data is None:
        data = CommentedMap()
    mutate(data)
    if path.exists():
        _backup(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        yaml.dump(data, f)
    os.replace(tmp, path)
    _reload()


def set_value(key: str, raw: str) -> str:
    """Store *raw* (coerced) at *key* in config.yaml, keeping comments and order, and reload
    the running config. Returns the confirmation text (the key, never the value)."""
    key = key.strip()
    check_key(key)
    value = coerce(key, raw)
    leaf = ConfigSchema.split_key(key)[-1]
    outcome = {}

    def mutate(data):
        node = _parent_node(data, key)
        if isinstance(leaf, int):
            # An item of a list of plain values: the index equal to the length appends. A taken
            # index is never overwritten — the LLM reads one listing and enables several items
            # from it, each with the slot number that was free when the listing was printed.
            parent = ConfigSchema.join_key(ConfigSchema.split_key(key)[:-1])
            if leaf > len(node):
                raise ConfigStoreError(f"{parent} has {len(node)} item(s) — [{len(node)}] adds the next one")
            if value in node:
                outcome["existed"] = True
                return
            if leaf < len(node):
                raise ConfigStoreError(f"{parent}[{leaf}] already holds another value — [{len(node)}] adds "
                                       f"the next one" + (", or use enable_integration" if parent == ENABLED_INTEGRATIONS_KEY else ""))
            outcome["existed"] = False
            node.append(value)
            return
        outcome["existed"] = leaf in node
        node[leaf] = value

    _rewrite(mutate)
    info(f"Config: set {key} ({'replaced' if outcome['existed'] else 'new'})")
    return f"{key} saved"


def _promoted(value: str, key: str) -> CommentedSeq:
    """A single value written by hand where a list of values belongs (`exclude_mics: Bose QC45`)
    becomes a one-item list, so the next write adds to it instead of being refused."""
    info(f"Config: {key} held a single value, promoted to a one-item list")
    return CommentedSeq([value])


def _enabled_list(data: CommentedMap) -> CommentedSeq:
    """config.yaml's `enabled_integrations` as a list to edit in place (a missing one is
    created, a single name is promoted to a one-item list)."""
    node = data.get(ENABLED_INTEGRATIONS_KEY)
    if node is None:
        node = data[ENABLED_INTEGRATIONS_KEY] = CommentedSeq()
    elif isinstance(node, str):
        node = data[ENABLED_INTEGRATIONS_KEY] = _promoted(node, ENABLED_INTEGRATIONS_KEY)
    if not isinstance(node, list):
        raise ConfigStoreError(f"{ENABLED_INTEGRATIONS_KEY} is not a list in config.yaml — set it by hand")
    return node


def enable_integration(name: str) -> str:
    """Add *name* to `enabled_integrations` by name: already listed under any spelling
    ('Sonos & Spotify', 'sonos-and-spotify') is a no-op, so no call can push another name out."""
    name = name.strip()
    if not name:
        raise ConfigStoreError("expected an integration name from list_integrations")
    wanted = integration_key(name)
    if wanted in {integration_key(n) for n in ConfigManager.get_instance().enabled_integrations()}:
        info(f"Config: {name} already in {ENABLED_INTEGRATIONS_KEY}, nothing to do")
        return f"{name} was already enabled"

    def mutate(data):
        node = _enabled_list(data)
        if wanted not in {integration_key(str(n)) for n in node}:
            node.append(name)

    _rewrite(mutate)
    info(f"Config: {name} added to {ENABLED_INTEGRATIONS_KEY}")
    return f"{name} enabled"


def disable_integration(name: str) -> str:
    """Remove every spelling of *name* from `enabled_integrations`."""
    name = name.strip()
    if not name:
        raise ConfigStoreError("expected an integration name from list_integrations")
    wanted = integration_key(name)
    if wanted not in {integration_key(n) for n in ConfigManager.get_instance().enabled_integrations()}:
        info(f"Config: {name} not in {ENABLED_INTEGRATIONS_KEY}, nothing to do")
        return f"{name} was not enabled"

    def mutate(data):
        node = _enabled_list(data)
        for i in range(len(node) - 1, -1, -1):
            if integration_key(str(node[i])) == wanted:
                del node[i]

    _rewrite(mutate)
    info(f"Config: {name} removed from {ENABLED_INTEGRATIONS_KEY}")
    return f"{name} disabled"


def rename_keys(renames: dict[str, str], integrations: Optional[dict[str, str]] = None) -> list[str]:
    """A release renamed config keys: move each *renames* old → new dotted key (a whole section
    or one entry of it, staying under the same parent) and each *integrations* old → new name in
    `enabled_integrations`. Order, comments and values stay; an old key whose new one already
    exists is left for the user to merge. Returns what was renamed; the file is only written
    when that is something."""
    done: list[str] = []

    def mutate(data):
        for old, new in renames.items():
            parent_path, _, old_leaf = old.rpartition(".")
            new_leaf = new.rpartition(".")[2]
            parent = data
            for part in filter(None, parent_path.split(".")):
                parent = parent.get(part) if isinstance(parent, dict) else None
            if not isinstance(parent, dict) or old_leaf not in parent:
                continue
            if new_leaf in parent:
                info(f"Config: both {old} and {new} exist — {old} left in place, merge it by hand")
                continue
            position = list(parent.keys()).index(old_leaf)
            comment = parent.ca.items.pop(old_leaf, None) if isinstance(parent, CommentedMap) else None
            parent.insert(position, new_leaf, parent.pop(old_leaf))
            if comment is not None:
                parent.ca.items[new_leaf] = comment
            done.append(f"{old} → {new}")
        node = data.get(ENABLED_INTEGRATIONS_KEY)
        for old, new in (integrations or {}).items():
            if not isinstance(node, list):
                break
            for i, listed in enumerate(node):
                if integration_key(str(listed)) == integration_key(old):
                    node[i] = new
                    done.append(f"{ENABLED_INTEGRATIONS_KEY}: {listed} → {new}")

    def present(key: str) -> bool:
        try:
            _current_value(key)
            return True
        except KeyError:
            return False

    listed = {integration_key(n) for n in ConfigManager.get_instance().enabled_integrations()}
    if not any(present(old) for old in renames) and not any(integration_key(old) in listed for old in integrations or {}):
        trace("Config: no renamed keys to migrate")
        return done
    _rewrite(mutate)
    return done


def _current_value(key: str) -> Any:
    """config.yaml's value at *key*, KeyError when absent — indexes into lists too."""
    parts = ConfigSchema.split_key(key)
    node: Any = ConfigManager.get_instance().get(parts[0])
    for part in parts[1:]:
        if isinstance(part, int):
            if not isinstance(node, list) or part >= len(node):
                raise KeyError(key)
        elif not isinstance(node, dict) or part not in node:
            raise KeyError(key)
        node = node[part]
    return node




def is_set(key: str) -> bool:
    """True when config.yaml holds a real value at *key*: present, and neither a copied
    example placeholder (YOUR_…, CHANGE_ME) nor the example's 0 for an id."""
    try:
        value = _current_value(key)
    except (KeyError, ValueError):
        return False
    if value is None or value == [] or value == {}:
        return False
    if looks_like_placeholder(value):
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, int) and value == 0 and ConfigSchema.example_value(key) == 0:
        return False
    return True


def expand(key: str) -> list[str]:
    """A section key ("email") as the leaf keys the example lists under it; a leaf as itself."""
    found = ConfigSchema.leaves(key)
    return found or [key]


def describe_key(key: str) -> dict:
    """`{"key", "set"}`; a list of sections (imap.accounts, icloud.devices) also carries a
    `summary` — "2 accounts, 1 without password" — since set / not set says little about a list."""
    described = {"key": key, "set": is_set(key)}
    if ConfigSchema.leaf_type(key) is list and not ConfigSchema.is_scalar_list(key):
        described["summary"] = _list_summary(key)
    return described


def _list_summary(key: str) -> str:
    """How many items config.yaml holds at a list of sections and how many lack a value the
    example marks as one to fill in (a placeholder such as YOUR_APP_PASSWORD) — by count and
    field name, never by value."""
    try:
        items = _current_value(key)
    except (KeyError, ValueError):
        items = None
    items = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
    noun = key.rsplit(".", 1)[-1].rstrip("s")
    text = f"{len(items)} {noun}{'s' if len(items) != 1 else ''}"
    shape = ConfigSchema.example_value(f"{key}[0]")
    required = [field for field, value in (shape or {}).items() if looks_like_placeholder(value)] if isinstance(shape, dict) else []
    for field in required:
        lacking = sum(1 for item in items if item.get(field) in (None, "") or looks_like_placeholder(item.get(field)))
        if lacking:
            text += f", {lacking} without {field}"
    trace(f"Config: {key} holds {text}")
    return text


def describe_keys(keys: list[str]) -> list[dict]:
    """One `describe_key` per leaf, sections expanded; the order of *keys* is kept."""
    out: list[dict] = []
    for key in keys:
        for leaf in expand(key):
            out.append(describe_key(leaf))
    return out


def summary(keys: list[str]) -> str:
    """'imap.accounts: 2 accounts, 1 without password, mail.send_from: not set' — the shape the assistant reads."""
    return ", ".join(f"{d['key']}: {d.get('summary') or ('set' if d['set'] else 'not set')}" for d in describe_keys(keys))


def section_summary(section: str) -> Optional[str]:
    """'2 of 3 set' for a section, None when the example has no leaves under it."""
    described = describe_keys([section]) if ConfigSchema.leaves(section) else []
    if not described:
        return None
    return f"{sum(1 for d in described if d['set'])} of {len(described)} set"
