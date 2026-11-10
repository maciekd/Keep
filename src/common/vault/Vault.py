"""The encrypted secret store, `vault.bin` (VAULT_PLAN §3.3).

File layout: a `FLOXYVAULT1` header line, a base64 16-byte salt line, then one Fernet
token whose plaintext is JSON Lines, one secret per line. The key is scrypt(passphrase,
salt). Writes go to `vault.bin.tmp` and are renamed into place, so the file is never
partial. A vault that cannot be read (no passphrase, wrong key, corrupt file) is
`enabled=False`, carries an `error`, and refuses every write — it never overwrites a
file it could not decrypt. A wrong key or a corrupt file is additionally `locked=True`,
which the prompt section and the redactor's logs report loudly: redaction is off then.
"""
import base64
import dataclasses
import json
import os
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from common.Paths import VAULT_PATH
from common.logging import info, error
from common.vault.SecretKinds import DEFAULT_COUNTRY_CODE, KINDS, VaultError, normalize

HEADER = b"FLOXYVAULT1"
MAX_ID_LENGTH = 40
# scrypt work factor. ~90 ms per key here; the unit conftest lowers it to 2**4 so a test
# vault costs microseconds (~300 of them made scrypt a third of the suite's time).
SCRYPT_N = 2 ** 14
_DOTS = "••••"

__all__ = ["Vault", "Secret", "VaultError", "DuplicateSecret", "KINDS", "HEADER"]


class DuplicateSecret(VaultError):
    """The value normalises to the same canonical as an existing entry."""

    def __init__(self, existing_id: str):
        super().__init__(f"same value as existing secret '{existing_id}'")
        self.existing_id = existing_id


@dataclass
class Secret:
    id: str
    kind: str
    description: str
    value: str
    created: str
    # Not persisted: the vault's default country code, needed to canonicalise phones.
    country_code: int = field(default=DEFAULT_COUNTRY_CODE, repr=False, compare=False)

    @property
    def canonical(self) -> str:
        return normalize(self.kind, self.value, self.country_code)

    @property
    def masked(self) -> str:
        if self.kind in ("phone", "card", "account"):
            return f"{_DOTS} {self.canonical[-4:]}"
        if self.kind == "email":
            local, _, domain = self.canonical.partition("@")
            return f"{local[:1]}•••@{domain}"
        if self.kind == "plate":
            return f"{self.canonical[:3]} •••"
        return _DOTS * 2

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "description": self.description,
                "value": self.value, "created": self.created}


def _slugify(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug[:MAX_ID_LENGTH].rstrip("-")


def _config_settings() -> tuple[Optional[str], int]:
    # Imported here so the vault core (and its tests) never need config.yaml.
    from common.ConfigManager import config
    passphrase = config.get("vault.key", None)
    country_code = int(config.get("vault.default_country_code", DEFAULT_COUNTRY_CODE))
    return (str(passphrase) if passphrase else None), country_code


class Vault:
    _instance: Optional["Vault"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "Vault":
        # The first callers arrive together at boot (the prompt section's thread and create_llm
        # on the main thread) and the constructor runs scrypt, so without the lock both could
        # build one — and the LLM wrapper holding the loser would never see a later edit.
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    passphrase, country_code = _config_settings()
                    cls._instance = cls(VAULT_PATH, passphrase, country_code)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        with cls._instance_lock:
            cls._instance = None

    def __init__(self, path, passphrase: Optional[str], country_code: int = DEFAULT_COUNTRY_CODE):
        self.path = Path(path)
        self.country_code = int(country_code)
        self.version = 0
        self.error: Optional[str] = None
        # True when there is a key but the file will not open with it (wrong key, corrupt file):
        # unlike a missing key, that is a misconfiguration the user must hear about.
        self.locked = False
        self._lock = threading.RLock()
        self._secrets: dict[str, Secret] = {}
        self._fernet: Optional[Fernet] = None
        self._salt: Optional[bytes] = None
        self._load(passphrase)

    @property
    def enabled(self) -> bool:
        return self.error is None

    # -- reading ---------------------------------------------------------------------

    def secrets(self) -> list[Secret]:
        with self._lock:
            return list(self._secrets.values())

    def get(self, secret_id: str) -> Optional[Secret]:
        with self._lock:
            return self._secrets.get(secret_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._secrets)

    def __contains__(self, secret_id: str) -> bool:
        with self._lock:
            return secret_id in self._secrets

    # -- writing ---------------------------------------------------------------------

    def add(self, kind: str, description: str, value: str, id: Optional[str] = None) -> Secret:
        with self._lock:
            self._require_writable()
            value = value.strip()
            canonical = normalize(kind, value, self.country_code)
            self._check_duplicate(canonical, exclude=None)
            secret_id = self._resolve_id(id, description, kind)
            secret = Secret(secret_id, kind, description.strip(), value,
                            datetime.now().isoformat(timespec="seconds"), self.country_code)
            self._commit(lambda: self._secrets.__setitem__(secret_id, secret))
            info(f"Vault: added '{secret_id}' ({kind}), {len(self._secrets)} secret(s) in {self.path.name}")
            return secret

    def update(self, secret_id: str, description: Optional[str] = None, value: Optional[str] = None,
               id: Optional[str] = None) -> Secret:
        """Change a secret's description, value and/or id; a None leaves that part alone. A new
        id is slugified and must be free, like `add`'s; the entry is re-keyed in one commit."""
        with self._lock:
            self._require_writable()
            secret = self._secrets.get(secret_id)
            if secret is None:
                raise VaultError(f"no secret '{secret_id}'")
            new_value = secret.value if value is None else value.strip()
            canonical = normalize(secret.kind, new_value, self.country_code)
            self._check_duplicate(canonical, exclude=secret_id)
            new_id = secret_id if id is None else self._resolve_id(id, secret.description, secret.kind, exclude=secret_id)
            updated = dataclasses.replace(
                secret, id=new_id, value=new_value,
                description=secret.description if description is None else description.strip())

            def mutate():
                if new_id != secret_id:
                    del self._secrets[secret_id]
                self._secrets[new_id] = updated

            self._commit(mutate)
            renamed = f" → '{new_id}'" if new_id != secret_id else ""
            info(f"Vault: updated '{secret_id}'{renamed}")
            return updated

    def suggest_id(self, description: str, kind: str, exclude: Optional[str] = None) -> str:
        """The id `add` would derive from `description` (the kind when it slugs to nothing, a
        `-N` suffix when taken). `exclude` is the secret about to be renamed: its own id is free."""
        with self._lock:
            return self._resolve_id(None, description, kind, exclude=exclude)

    def delete(self, secret_id: str) -> None:
        with self._lock:
            self._require_writable()
            if secret_id not in self._secrets:
                raise VaultError(f"no secret '{secret_id}'")
            self._commit(lambda: self._secrets.pop(secret_id))
            info(f"Vault: deleted '{secret_id}', {len(self._secrets)} secret(s) left")

    # -- internals -------------------------------------------------------------------

    def _require_writable(self):
        if not self.enabled:
            error(f"Vault: write refused — {self.error}")
            raise VaultError(self.error)

    def _check_duplicate(self, canonical: str, exclude: Optional[str]):
        for secret in self._secrets.values():
            if secret.id != exclude and secret.canonical == canonical:
                raise DuplicateSecret(secret.id)

    def _resolve_id(self, given: Optional[str], description: str, kind: str,
                    exclude: Optional[str] = None) -> str:
        taken = {i for i in self._secrets if i != exclude}
        if given:
            slug = _slugify(given)
            if not slug:
                raise VaultError(f"invalid id '{given}'")
            if slug in taken:
                raise VaultError(f"id '{slug}' already exists")
            return slug
        base = _slugify(description) or kind.replace("_", "-")
        candidate, n = base, 1
        while candidate in taken:
            n += 1
            suffix = f"-{n}"
            candidate = base[:MAX_ID_LENGTH - len(suffix)].rstrip("-") + suffix
        return candidate

    def _commit(self, mutate: Callable[[], object]):
        before = dict(self._secrets)
        mutate()
        try:
            self._save()
        except Exception:
            self._secrets = before
            raise
        self.version += 1

    @staticmethod
    def _fernet_for(passphrase: str, salt: bytes) -> Fernet:
        key = Scrypt(salt=salt, length=32, n=SCRYPT_N, r=8, p=1).derive(passphrase.encode())
        return Fernet(base64.urlsafe_b64encode(key))

    def _load(self, passphrase: Optional[str]):
        if not passphrase:
            self.error = "vault.key not set in config.yaml"
            info(f"Vault: {self.error} — vault disabled")
            return
        if not self.path.exists():
            self._salt = os.urandom(16)
            self._fernet = self._fernet_for(passphrase, self._salt)
            info(f"Vault: no {self.path.name} yet — starting empty")
            return
        try:
            header, salt_b64, token = self.path.read_bytes().split(b"\n", 2)
            if header != HEADER:
                raise VaultError(f"{self.path.name} is not a vault file")
            salt = base64.b64decode(salt_b64, validate=True)
            fernet = self._fernet_for(passphrase, salt)
            plaintext = fernet.decrypt(token.strip()).decode()
            secrets = [Secret(**json.loads(line), country_code=self.country_code)
                       for line in plaintext.splitlines() if line.strip()]
        except InvalidToken:
            self._lock_out(f"wrong vault.key or corrupt {self.path.name}")
            return
        except (ValueError, TypeError) as e:
            self._lock_out(f"cannot read {self.path.name}: {e}")
            return
        self._salt, self._fernet = salt, fernet
        self._secrets = {s.id: s for s in secrets}
        info(f"Vault: loaded {len(self._secrets)} secret(s) from {self.path}")

    def _lock_out(self, message: str):
        self.error = message
        self.locked = True
        self._fernet = None
        error(f"Vault: locked — {message}; all writes refused")

    def _save(self):
        lines = "".join(json.dumps(s.to_dict(), ensure_ascii=False) + "\n" for s in self._secrets.values())
        token = self._fernet.encrypt(lines.encode())
        data = HEADER + b"\n" + base64.b64encode(self._salt) + b"\n" + token
        tmp = self.path.with_name(self.path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        os.chmod(self.path, 0o600)
