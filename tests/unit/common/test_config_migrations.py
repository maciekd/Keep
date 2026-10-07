"""ConfigMigrations: the move of the `email`, `icloud_email` and `calendar` sections into the
accounts lists, run on temp copies — each mapping, the enabled-integrations rename, the
commented-out originals, idempotence, and a file without the old sections left untouched."""
import os

import pytest
import yaml

from common import ConfigMigrations
from common.ConfigManager import ConfigManager

OLD = """\
# the household's config
enabled_integrations: [Gmail, Tesla, iCloud Email, CalDAV]

api_keys:
  openai: sk-fictional   # paid

email:
  address: marcus@example.com   # the gmail
  password: gmail-app-pw
  smtp_host: smtp.example
  smtp_port: 2525
icloud_email:
  address: m@mlab.test
  password: icloud-app-pw

calendar:
  username: m@mlab.test
  password: caldav-app-pw
  default_calendar: Family
  shown_calendars: [Home, Family]

vault:
  key: real-passphrase
"""


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ConfigMigrations, "BACKUP_DIR", tmp_path / "backups")
    ConfigMigrations.take_note()
    path = tmp_path / "config.yaml"
    path.write_text(OLD)
    return path


def _load(path):
    return yaml.safe_load(path.read_text())


@pytest.mark.unit
class TestMappings:
    def test_gmail_and_icloud_mail_become_imap_accounts_with_their_presets_a_contract_with_imap_accounts(self, config_file):
        ConfigMigrations.run(str(config_file))
        gmail, icloud = _load(config_file)["imap"]["accounts"]
        assert gmail == {"name": "Gmail", "preset": "gmail", "address": "marcus@example.com", "password": "gmail-app-pw",
                         "smtp_host": "smtp.example", "smtp_port": 2525}
        assert icloud == {"name": "iCloud", "preset": "icloud", "address": "m@mlab.test", "password": "icloud-app-pw"}

    def test_the_calendar_section_becomes_the_first_caldav_account_a_contract_with_caldav_accounts(self, config_file):
        ConfigMigrations.run(str(config_file))
        account, = _load(config_file)["caldav"]["accounts"]
        assert account == {"name": "iCloud", "preset": "icloud", "username": "m@mlab.test", "password": "caldav-app-pw",
                           "default_calendar": "Family", "shown_calendars": ["Home", "Family"]}

    def test_a_calendar_without_a_default_keeps_home_the_default_it_had(self, tmp_path, config_file):
        config_file.write_text("calendar:\n  username: m@mlab.test\n  password: pw\n")
        ConfigMigrations.run(str(config_file))
        account, = _load(config_file)["caldav"]["accounts"]
        assert account["default_calendar"] == "Home" and "shown_calendars" not in account

    def test_the_old_names_in_enabled_integrations_become_imap_mail_once_in_the_first_ones_place(self, config_file):
        ConfigMigrations.run(str(config_file))
        assert _load(config_file)["enabled_integrations"] == ["IMAP Mail", "Tesla", "CalDAV"]

    def test_an_already_listed_imap_mail_absorbs_the_old_names(self, config_file):
        config_file.write_text("enabled_integrations: [IMAP Mail, gmail, Tesla]\n")
        ConfigMigrations.run(str(config_file))
        assert _load(config_file)["enabled_integrations"] == ["IMAP Mail", "Tesla"]

    def test_send_from_is_icloud_when_an_icloud_mail_account_was_migrated(self, config_file):
        ConfigMigrations.run(str(config_file))
        assert _load(config_file)["mail"]["send_from"] == "iCloud"

    def test_send_from_is_left_alone_without_an_icloud_account_or_when_already_set(self, config_file):
        config_file.write_text("email:\n  address: marcus@example.com\n  password: pw\n")
        ConfigMigrations.run(str(config_file))
        assert "mail" not in _load(config_file)
        config_file.write_text("mail:\n  send_from: Work\nicloud_email:\n  address: m@mlab.test\n  password: pw\n")
        ConfigMigrations.run(str(config_file))
        assert _load(config_file)["mail"]["send_from"] == "Work"

    def test_an_existing_accounts_list_is_appended_to(self, config_file):
        config_file.write_text("imap:\n  accounts:\n    - name: Work\n      preset: gmail\nemail:\n  address: marcus@example.com\n  password: pw\n")
        ConfigMigrations.run(str(config_file))
        assert [a["name"] for a in _load(config_file)["imap"]["accounts"]] == ["Work", "Gmail"]

    def test_a_partly_filled_section_carries_only_its_real_values_over(self, config_file):
        """A placeholder password is screened out like an empty one, but the address beside it
        is real, so an account is made — without the password key, which the owner fills in."""
        config_file.write_text("email:\n  address: m@mlab.test\n  password: YOUR_APP_PASSWORD\n  smtp_host: smtp.example\n"
                               "calendar:\n  username: m@mlab.test\n  password: CHANGE_ME\n")
        changes = ConfigMigrations.run(str(config_file))
        data = _load(config_file)
        gmail, = data["imap"]["accounts"]
        assert gmail == {"name": "Gmail", "preset": "gmail", "address": "m@mlab.test", "smtp_host": "smtp.example"}
        caldav, = data["caldav"]["accounts"]
        assert caldav == {"name": "iCloud", "preset": "icloud", "username": "m@mlab.test", "default_calendar": "Home"}
        assert any("imap.accounts[0]" in c for c in changes) and any("caldav.accounts[0]" in c for c in changes)

    def test_a_scalar_enabled_integrations_becomes_a_list_with_imap_mail(self, config_file):
        config_file.write_text("enabled_integrations: Gmail\nemail:\n  address: m@mlab.test\n  password: pw\n")
        ConfigMigrations.run(str(config_file))
        assert _load(config_file)["enabled_integrations"] == [ConfigMigrations.IMAP_INTEGRATION]

        config_file.write_text("enabled_integrations: iCloud Email\n")
        assert ConfigMigrations.run(str(config_file))
        assert _load(config_file)["enabled_integrations"] == [ConfigMigrations.IMAP_INTEGRATION]

    def test_send_from_joins_an_existing_mail_section_without_disturbing_its_other_keys(self, config_file):
        config_file.write_text("mail:\n  signature: Marcus\n  max_results: 20\nicloud_email:\n  address: m@mlab.test\n  password: pw\n")
        ConfigMigrations.run(str(config_file))
        assert _load(config_file)["mail"] == {"signature": "Marcus", "max_results": 20, "send_from": "iCloud"}


@pytest.mark.unit
class TestTheFile:
    def test_the_old_sections_stay_commented_out_with_the_date_and_every_other_comment_survives(self, config_file):
        ConfigMigrations.run(str(config_file))
        text = config_file.read_text()
        assert text.startswith("# the household's config\n") and "# paid" in text
        assert "# migrated 20" in text
        assert "# email:\n#   address: marcus@example.com   # the gmail" in text
        assert "# calendar:\n#   username: m@mlab.test" in text
        assert "email" not in _load(config_file) and "calendar" not in _load(config_file)

    def test_the_second_run_finds_nothing_and_leaves_the_file_as_it_was(self, config_file):
        assert ConfigMigrations.run(str(config_file))
        after_first = config_file.read_text()
        assert ConfigMigrations.run(str(config_file)) == []
        assert config_file.read_text() == after_first

    def test_a_never_filled_section_is_commented_out_but_makes_no_account(self, config_file):
        """An install that copied the old example and never filled it in: the placeholders are
        not carried over and no half account is made, while the rename in enabled_integrations
        and the commented-out originals still happen — so the second run finds nothing."""
        config_file.write_text("enabled_integrations: [Gmail, CalDAV]\n"
                               "email:\n  address: YOUR_GMAIL_ADDRESS\n  password: YOUR_APP_PASSWORD\n"
                               "calendar: {}\n")
        changes = ConfigMigrations.run(str(config_file))
        data = _load(config_file)
        assert data["imap"]["accounts"] == [] and data["caldav"]["accounts"] == []
        assert data["enabled_integrations"] == ["IMAP Mail", "CalDAV"]
        assert "mail" not in data
        assert "# email:" in config_file.read_text() and "# calendar:" in config_file.read_text()
        assert len(changes) == 3 and ConfigMigrations.run(str(config_file)) == []

    def test_a_file_without_the_old_sections_is_not_even_rewritten(self, config_file):
        config_file.write_text("enabled_integrations: [Tesla, IMAP Mail]\nimap:\n  accounts: []\n")
        os.utime(config_file, (1_000_000, 1_000_000))
        assert ConfigMigrations.run(str(config_file)) == []
        assert os.stat(config_file).st_mtime == 1_000_000 and not (config_file.parent / "backups").exists()

    def test_a_backup_is_taken_before_the_rewrite(self, config_file):
        ConfigMigrations.run(str(config_file))
        backup, = (config_file.parent / "backups").iterdir()
        assert backup.read_text() == OLD

    def test_the_note_summarises_the_moves_and_is_taken_once(self, config_file):
        changes = ConfigMigrations.run(str(config_file))
        note = ConfigMigrations.take_note()
        assert note and all(change in note for change in changes) and "imap.accounts" in note
        assert ConfigMigrations.take_note() is None

    def test_a_file_that_is_not_yaml_is_left_alone(self, config_file):
        config_file.write_text("email: [unclosed\n")
        assert ConfigMigrations.run(str(config_file)) == []


@pytest.mark.unit
class TestOnLoad:
    def test_config_manager_migrates_on_load_and_serves_the_new_sections(self, config_file, monkeypatch):
        monkeypatch.setenv("FLOXY_CONFIG", str(config_file))
        manager = ConfigManager()
        assert [a["name"] for a in manager.get("imap.accounts")] == ["Gmail", "iCloud"]
        assert manager.enabled_integrations() == ["IMAP Mail", "Tesla", "CalDAV"]
        assert "# migrated" in config_file.read_text()
        assert ConfigMigrations.take_note()

    def test_a_config_without_the_old_sections_is_read_without_a_rewrite(self, tmp_path, monkeypatch):
        path = tmp_path / "config.yaml"
        path.write_text("vault:\n  key: real\n")
        os.utime(path, (1_000_000, 1_000_000))
        monkeypatch.setenv("FLOXY_CONFIG", str(path))
        assert ConfigManager().get("vault.key") == "real"
        assert os.stat(path).st_mtime == 1_000_000


def test_the_launchers_import_chain_survives_a_config_that_needs_migrating(tmp_path):
    """The launcher imports Migrations → ConfigStore → ConfigManager, whose module-level instance
    loads the config and migrates it while ConfigStore is still half-imported (2026-10-07: the
    production start died there on a circular import). A fresh interpreter is the only way to
    see that order."""
    import subprocess, sys
    from common.Paths import AISERVER_DIR, SRC_DIR
    cfg = tmp_path / "config.yaml"
    cfg.write_text("enabled_integrations: [Gmail]\nemail:\n  address: m@mlab.test\n  password: pw\n")
    code = "from common import ConfigMigrations, Migrations, Version; from common.ConfigManager import config; print(sorted(config.get('imap')['accounts'][0]))"
    result = subprocess.run([sys.executable, "-c", code], cwd=AISERVER_DIR, capture_output=True, text=True, timeout=60,
                            env={**__import__('os').environ, "FLOXY_CONFIG": str(cfg), "PYTHONPATH": str(SRC_DIR)})
    assert result.returncode == 0, result.stderr[-2000:]
    assert "['address', 'name', 'password', 'preset']" in result.stdout
