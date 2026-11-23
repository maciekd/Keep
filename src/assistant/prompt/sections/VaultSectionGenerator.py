from pathlib import Path

from ..PromptSectionGenerator import PromptSectionGenerator
from common.Paths import MEMORY_DIR
from common.logging import trace, info
from common.vault.Redactor import placeholder
from common.vault.Vault import Vault


class VaultSectionGenerator(PromptSectionGenerator):
    """Lists the vault's orphan secrets as `{{secret:id}}` placeholders — id, description and kind, never a value.
    An orphan is a secret no file under `memory/` mentions: the bank, a watch item, a procedure or a memo names the
    others with their context ("Marcus's Amex: {{secret:amex}}"), so repeating them here would only cost
    tokens. Re-renders as soon as `vault.version` moves or a memory file changes, so a TUI edit or a
    `remember` is live for the next request. Rendered at the end of the prompt's `## Vault` subsection,
    whose prose introduces the list; no heading of its own."""

    LOCKED = ("The vault is LOCKED: {error}. Its secrets cannot be read, so there are no placeholders and "
              "nothing is hidden from you. Tell the user, so they can fix vault.key in config.yaml.")

    def __init__(self, prompt_manager, vault: Vault | None = None, memory_dir: Path = MEMORY_DIR):
        super().__init__(prompt_manager, "Vault")
        # Resolved lazily so constructing a PromptManager never reads config.yaml / vault.bin.
        self._vault = vault
        self.memory_dir = Path(memory_dir)
        self.last_version: int | None = None
        self.last_memory_stamp: tuple | None = None

    @property
    def vault(self) -> Vault:
        if self._vault is None:
            self._vault = Vault.get_instance()
        return self._vault

    def _start_monitoring(self):
        """Check the vault's version counter and the memory files' stamps every second."""
        while not self._stop_event.is_set():
            try:
                self._refresh_if_changed()
            except Exception as e:
                trace(f"Error checking vault section: {e}")
            self._wait(1)

    def _refresh_if_changed(self):
        # Read the version and the stamp before rendering: an edit that lands mid-render moves them
        # again, so the next tick re-renders instead of being skipped.
        version = self.vault.version
        stamp = self._memory_stamp()
        if version != self.last_version or stamp != self.last_memory_stamp:
            self._update_content()
            self.last_version = version
            self.last_memory_stamp = stamp

    def _memory_files(self) -> list[Path]:
        if not self.memory_dir.is_dir():
            return []
        return sorted(self.memory_dir.rglob("*.md"))

    def _memory_stamp(self) -> tuple:
        """What changes when a memory file is written: a cheap stat per file, no reading."""
        stamp = []
        for path in self._memory_files():
            try:
                st = path.stat()
            except OSError:
                continue
            stamp.append((str(path), st.st_mtime_ns, st.st_size))
        return tuple(stamp)

    def _memory_text(self) -> str:
        texts = []
        for path in self._memory_files():
            try:
                texts.append(path.read_text())
            except OSError as e:
                trace(f"Vault section: cannot read {path}: {e}")
        return "\n".join(texts)

    def _fetch_content(self) -> str:
        if self.vault.locked:
            # Fail-open must not be silent: with the vault locked every request goes out
            # unredacted, and this is how the user gets to hear about it.
            info(f"Vault section: vault LOCKED ({self.vault.error}) — the prompt says so")
            return self.LOCKED.format(error=self.vault.error)
        secrets = sorted(self.vault.secrets(), key=lambda s: s.id)
        if not secrets:
            state = "empty" if self.vault.enabled else f"disabled ({self.vault.error})"
            info(f"Vault section: vault {state}")
            return "The vault is empty."
        memory = self._memory_text()
        orphans = [s for s in secrets if placeholder(s.id) not in memory]
        named = len(secrets) - len(orphans)
        info(f"Vault section: {len(secrets)} secret(s), {named} named in memory, {len(orphans)} listed")
        trace("Vault section orphan ids: " + ", ".join(s.id for s in orphans))
        if not orphans:
            return f"All {len(secrets)} placeholders are named in your memory; there are no others."
        lines = ["The placeholders your memory does not name, each with its description and kind:"]
        lines += [f"- {placeholder(s.id)} — {s.description} ({s.kind})" for s in orphans]
        if named:
            lines.append(f"Your memory names the other {named}.")
        return "\n".join(lines)
