"""What the memory stores write to disk carries vault placeholders, never the values:
`VaultLLM` restores placeholders across the whole reply, so the text these functions get
from a reply holds the real card number."""
import pytest

from assistant import MemoryFile, WatchList
from assistant.Procedures import Procedures
from common.vault import Redactor as redactor_module
from common.vault.Redactor import placeholder
from common.vault.Vault import Vault

CARD = "4111 1111 1111 1111"
PLACEHOLDER = placeholder("marcus-visa")


@pytest.fixture
def stocked_singleton(tmp_path, monkeypatch):
    vault = Vault(tmp_path / "vault.bin", "correct horse battery staple")
    vault.add("card", "Marcus's Visa", CARD, id="marcus-visa")
    monkeypatch.setattr(Vault, "_instance", vault)
    monkeypatch.setattr(redactor_module, "_storage_redactor", None)
    return vault


@pytest.mark.unit
class TestStoresKeepPlaceholders:
    def test_a_saved_procedure_holds_the_placeholder(self, stocked_singleton, tmp_path):
        pd, log = tmp_path / "procedures", tmp_path / "log"
        assert Procedures.save("shop", "buying things", f"pay with {CARD} first", pd, log) == ""
        on_disk = (pd / "shop.md").read_text()
        assert PLACEHOLDER in on_disk and CARD not in on_disk

    def test_a_procedure_note_holds_the_placeholder(self, stocked_singleton, tmp_path):
        pd, log = tmp_path / "procedures", tmp_path / "log"
        Procedures.save("shop", "buying things", "steps", pd, log)
        assert Procedures.note("shop", f"the card {CARD} worked", pd) == ""
        on_disk = (pd / "shop.md").read_text()
        assert PLACEHOLDER in on_disk and CARD not in on_disk

    def test_a_watch_item_holds_the_placeholder(self, stocked_singleton, tmp_path):
        wl, log = tmp_path / "watchlist", tmp_path / "log"
        item = WatchList.save("Parcel", f"paid with {CARD}", None, wl, log)
        on_disk = (wl / f"{item.name}.md").read_text()
        assert PLACEHOLDER in on_disk and CARD not in on_disk
        assert WatchList.get("Parcel", wl).note == f"paid with {PLACEHOLDER}"

    def test_the_bank_holds_the_placeholder_on_append_and_rewrite(self, stocked_singleton, tmp_path):
        bank = str(tmp_path / "bank.md")
        MemoryFile.append(f"* card {CARD}", bank)
        assert CARD not in MemoryFile.read(bank) and PLACEHOLDER in MemoryFile.read(bank)
        MemoryFile.write(f"# Bank\n* curated card {CARD}\n", bank)
        on_disk = MemoryFile.read(bank)
        assert PLACEHOLDER in on_disk and CARD not in on_disk
