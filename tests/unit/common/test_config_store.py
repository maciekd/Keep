import threading

import pytest
import yaml

from common import ConfigSchema, ConfigStore
from common.ConfigManager import ConfigManager
from common.ConfigStore import ConfigStoreError

EXAMPLE = """\
enabled_integrations: []
api_keys:
  openai: YOUR_OPENAI_API_KEY_HERE
  grok: YOUR_GROK_API_KEY_HERE

local_outposts:
  enabled: true
  exclude_mics: []          # name substrings of mics to skip

pager:
  address: your_pager_address@example.com
  password: YOUR_PAGER_APP_PASSWORD  # Generate at https://pager.example/apppasswords

telegram_feed:
  api_id: 0          # Get from https://my.telegram.org/apps
  api_hash: YOUR_TELEGRAM_API_HASH_HERE

tesla:
  tessie_access_token: YOUR_TESSIE_ACCESS_TOKEN
  vehicles:
    - name: home_car
      vin: YOUR_TESLA_VIN

#photos:
#  videos:
#    enabled: true
#    max_size_mb: 150

icloud:
  apple_id: your_apple_id@icloud.com
  devices:
    - id: "DEVICE_UUID_FROM_ICLOUD_TAB"
      display_name: "dad"

chat:
  secret_request_timeout_seconds: 120

interfaces:
  voice:    { enabled: false }
  console:  { enabled: true, port: 5151 }

vault:
  key: CHANGE_ME
  default_country_code: 46

location:
  home:
    latitude: 34.67
owner_name: YOUR_NAME
"""

CONFIG = """\
# my keys
api_keys:
  openai: sk-real-openai            # paid account
  grok: YOUR_GROK_API_KEY_HERE

telegram_feed:
  api_id: 0
  api_hash: YOUR_TELEGRAM_API_HASH_HERE

tesla:
  tessie_access_token: tok-real
  vehicles:
    - name: home_car
      vin: 5YJ3E1EA7KF000001

vault:
  key: real-passphrase
  default_country_code: 46

location:
  home:
    latitude: 59.3
owner_name: Marcus
"""


@pytest.fixture(autouse=True)
def files(tmp_path, monkeypatch):
    example = tmp_path / "config.example.yaml"
    example.write_text(EXAMPLE)
    monkeypatch.setattr(ConfigSchema, "EXAMPLE_PATH", example)
    ConfigSchema.reset()
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG)
    monkeypatch.setenv("FLOXY_CONFIG", str(config))
    monkeypatch.setattr(ConfigStore, "BACKUP_DIR", tmp_path / "backups")
    ConfigManager.reset_instance()
    yield config
    ConfigManager.reset_instance()
    ConfigSchema.reset()
    # The module-level `config` was reloaded from the temp file: point it back at the real one.
    monkeypatch.delenv("FLOXY_CONFIG")
    from common import ConfigManager as module
    module.config.reload()


def _load(path):
    return yaml.safe_load(path.read_text())


@pytest.mark.unit
class TestSetValue:
    def test_replaces_a_value_and_keeps_every_other_line(self, files):
        before = files.read_text().splitlines()
        assert ConfigStore.set_value("api_keys.grok", "xai-real") == "api_keys.grok saved"
        after = files.read_text().splitlines()
        changed = [(a, b) for a, b in zip(before, after) if a != b]
        assert len(before) == len(after)
        assert changed == [("  grok: YOUR_GROK_API_KEY_HERE", "  grok: xai-real")]
        assert "# paid account" in files.read_text()

    def test_creates_a_missing_section_and_key(self, files):
        ConfigStore.set_value("pager.address", "m@mlab.test")
        ConfigStore.set_value("pager.password", "abcd efgh")
        data = _load(files)
        assert data["pager"] == {"address": "m@mlab.test", "password": "abcd efgh"}
        assert data["owner_name"] == "Marcus"

    def test_running_config_sees_the_new_value(self, files):
        ConfigStore.set_value("owner_name", "Anna")
        assert ConfigManager.get_instance().get("owner_name") == "Anna"

    def test_the_assistant_can_switch_an_interface_on(self, files):
        assert ConfigStore.set_value("interfaces.voice.enabled", "true") == "interfaces.voice.enabled saved"
        assert ConfigStore.set_value("interfaces.console.enabled", "false") == "interfaces.console.enabled saved"
        data = _load(files)
        assert data["interfaces"]["voice"] == {"enabled": True}
        assert data["interfaces"]["console"]["enabled"] is False

    def test_coerces_to_the_example_type(self, files):
        ConfigStore.set_value("telegram_feed.api_id", " 12345 ")
        ConfigStore.set_value("location.home.latitude", "57.7")
        ConfigStore.set_value("photos.videos.enabled", "no")
        data = _load(files)
        assert data["telegram_feed"]["api_id"] == 12345
        assert data["location"]["home"]["latitude"] == 57.7
        assert data["photos"]["videos"]["enabled"] is False

    @pytest.mark.parametrize("key,raw,reason", [
        ("telegram_feed.api_id", "abc", "expects a int"),
        ("location.home.latitude", "north", "expects a float"),
        ("photos.videos.enabled", "maybe", "expects a bool"),
        ("owner_name", "   ", "empty"),
    ])
    def test_rejects_a_value_of_the_wrong_shape(self, files, key, raw, reason):
        before = files.read_text()
        with pytest.raises(ConfigStoreError, match=reason):
            ConfigStore.set_value(key, raw)
        assert files.read_text() == before

    @pytest.mark.parametrize("key,reason", [
        ("vault.key", "cannot be changed"),
        ("interfaces.http.enabled", "cannot be changed"),
        ("indigo.log_refiner", "cannot be changed"),
        ("indigo.ssh_user", "cannot be changed"),
        ("photos.screen_prepare_command", "cannot be changed"),
        ("indigo.prompt_file", "cannot be changed"),
        ("vault.key[0]", "cannot be changed"),              # an indexed spelling still changes the key
        ("indigo.ssh_user[0].host", "cannot be changed"),
        ("interfaces", "not a single value"),
        ("gmail.password", "not a known setting"),
        ("pager/password", "not a known setting"),
        ("tesla.vehicles", "is a list — address one item's field"),
        ("api_keys", "not a single value"),
    ])
    def test_refuses_denied_unknown_and_structured_keys(self, files, key, reason):
        with pytest.raises(ConfigStoreError, match=reason):
            ConfigStore.check_key(key)
        with pytest.raises(ConfigStoreError, match=reason):
            ConfigStore.set_value(key, "x")

    def test_a_key_the_example_lacks_under_a_known_section_is_stored_as_text(self, files):
        ConfigStore.set_value("pager.imap_host", "imap.example.org")
        assert _load(files)["pager"]["imap_host"] == "imap.example.org"

    def test_backs_up_before_writing_and_keeps_only_the_newest(self, files, monkeypatch):
        monkeypatch.setattr(ConfigStore, "BACKUPS_KEPT", 2)
        stamps = iter(["20260908-100000", "20260908-100001", "20260908-100002"])
        monkeypatch.setattr(ConfigStore, "_stamp", lambda: next(stamps))
        ConfigStore.set_value("owner_name", "A")
        ConfigStore.set_value("owner_name", "B")
        ConfigStore.set_value("owner_name", "C")
        backups = sorted(p.name for p in ConfigStore.BACKUP_DIR.glob("config-*.yaml"))
        assert backups == ["config-20260908-100001.yaml", "config-20260908-100002.yaml"]
        assert "owner_name: B" in (ConfigStore.BACKUP_DIR / "config-20260908-100002.yaml").read_text()
        assert not files.with_suffix(".yaml.tmp").exists()

    def test_backups_within_one_second_do_not_overwrite_each_other(self, files):
        # The config lane writes several keys back to back; the backup taken before the first
        # one must survive the rest.
        ConfigStore.set_value("owner_name", "A")
        ConfigStore.set_value("owner_name", "B")
        backups = sorted(ConfigStore.BACKUP_DIR.glob("config-*.yaml"))
        assert len(backups) == 2
        assert "owner_name: Marcus" in backups[0].read_text()
        assert "owner_name: A" in backups[1].read_text()

    def test_stamp_has_microseconds_and_sorts_chronologically(self, monkeypatch):
        monkeypatch.setattr(ConfigStore.time, "time_ns", lambda: 1788000000_000123_000)
        assert ConfigStore._stamp() == "20260829-104000-000123"
        monkeypatch.setattr(ConfigStore.time, "time_ns", lambda: 1788000000_999999_000)
        assert ConfigStore._stamp() == "20260829-104000-999999"

    def test_logs_the_key_but_never_the_value(self, files, capsys):
        ConfigStore.set_value("pager.password", "hunter2-app-pw")
        out = capsys.readouterr().out
        assert "Config: set pager.password (new)" in out
        assert "hunter2" not in out
        ConfigStore.set_value("pager.password", "hunter3-app-pw")
        out = capsys.readouterr().out
        assert "Config: set pager.password (replaced)" in out
        assert "hunter3" not in out


@pytest.mark.unit
class TestDescribe:
    def test_set_means_present_and_not_a_placeholder(self, files):
        assert ConfigStore.is_set("api_keys.openai")
        assert not ConfigStore.is_set("api_keys.grok")          # copied YOUR_…_HERE
        assert not ConfigStore.is_set("pager.address")          # absent
        assert not ConfigStore.is_set("telegram_feed.api_id")  # the example's 0
        assert ConfigStore.is_set("vault.default_country_code")   # a real number
        assert ConfigStore.is_set("owner_name")

    def test_lists_are_set_when_they_hold_no_placeholder(self, files):
        assert ConfigStore.is_set("tesla.vehicles")
        ConfigStore.set_value("owner_name", "x")  # any write, to exercise reload before the next read
        files.write_text(files.read_text().replace("5YJ3E1EA7KF000001", "YOUR_TESLA_VIN"))
        ConfigManager.get_instance().reload()
        assert not ConfigStore.is_set("tesla.vehicles")

    def test_summary_expands_sections_to_their_leaves(self, files):
        assert ConfigStore.summary(["pager"]) == "pager.address: not set, pager.password: not set"
        assert ConfigStore.summary(["tesla"]) == "tesla.tessie_access_token: set, tesla.vehicles: 1 vehicle"
        assert ConfigStore.summary(["api_keys.grok", "owner_name"]) == "api_keys.grok: not set, owner_name: set"

    def test_a_key_outside_the_example_is_reported_by_itself(self, files):
        assert ConfigStore.summary(["reminders"]) == "reminders: not set"

    def test_section_summary_counts(self, files):
        assert ConfigStore.section_summary("api_keys") == "1 of 2 set"
        assert ConfigStore.section_summary("pager") == "0 of 2 set"
        assert ConfigStore.section_summary("owner_name") == "1 of 1 set"
        assert ConfigStore.section_summary("nope") is None

    def test_describe_never_returns_values(self, files):
        described = ConfigStore.describe_keys(["api_keys", "vault"])
        assert all(set(d) == {"key", "set"} for d in described)

    def test_a_list_of_sections_is_summarised_by_count_and_by_the_placeholder_fields_still_to_fill(self, files):
        """The example marks a device's id as a placeholder to replace (DEVICE_UUID…), so the
        summary counts the devices that still lack one — by count and field, never by value."""
        ConfigStore.set_value("icloud.devices[0].display_name", "dad")
        ConfigStore.set_value("icloud.devices[1].display_name", "mum")
        ConfigStore.set_value("icloud.devices[1].id", "7f1a-fictional-device")

        described, = ConfigStore.describe_keys(["icloud.devices"])

        assert described["summary"] == "2 devices, 1 without id"
        assert described["set"] is True
        assert "dad" not in str(described) and "7f1a" not in str(described)
        assert ConfigStore.summary(["icloud.devices"]) == "icloud.devices: 2 devices, 1 without id"


@pytest.mark.unit
class TestIndexedListKeys:
    def test_replaces_one_field_of_an_existing_item(self, files):
        assert ConfigStore.set_value("tesla.vehicles[0].vin", "VIN-NEW") == "tesla.vehicles[0].vin saved"
        assert _load(files)["tesla"]["vehicles"] == [{"name": "home_car", "vin": "VIN-NEW"}]
        assert "# my keys" in files.read_text()

    def test_index_equal_to_the_length_adds_an_item(self, files):
        ConfigStore.set_value("tesla.vehicles[1].name", "van")
        ConfigStore.set_value("tesla.vehicles[1].vin", "VIN2")
        assert _load(files)["tesla"]["vehicles"] == [
            {"name": "home_car", "vin": "5YJ3E1EA7KF000001"}, {"name": "van", "vin": "VIN2"}]

    def test_creates_the_section_and_list_when_missing(self, files):
        ConfigStore.set_value("icloud.devices[0].id", "uuid-1")
        ConfigStore.set_value("icloud.devices[0].display_name", "dad")
        assert _load(files)["icloud"] == {"devices": [{"id": "uuid-1", "display_name": "dad"}]}
        assert ConfigManager.get_instance().get("icloud.devices") == [{"id": "uuid-1", "display_name": "dad"}]

    def test_an_index_past_the_end_is_refused(self, files):
        with pytest.raises(ConfigStoreError, match=r"tesla.vehicles has 1 item\(s\) — \[1\] adds the next one"):
            ConfigStore.set_value("tesla.vehicles[2].vin", "VIN3")
        with pytest.raises(ConfigStoreError, match=r"icloud.devices has 0 item\(s\) — \[0\] adds the next one"):
            ConfigStore.set_value("icloud.devices[1].id", "uuid")

    def test_a_list_or_a_whole_item_is_not_a_single_value(self, files):
        with pytest.raises(ConfigStoreError, match=r"is a list — address one item's field, e.g. icloud.devices\[0\].<field>"):
            ConfigStore.check_key("icloud.devices")
        with pytest.raises(ConfigStoreError, match="not a single value"):
            ConfigStore.check_key("icloud.devices[0]")

    def test_unknown_fields_of_an_item_are_allowed_like_any_unlisted_key(self, files):
        ConfigStore.set_value("tesla.vehicles[0].display_name", "Family car")
        assert _load(files)["tesla"]["vehicles"][0]["display_name"] == "Family car"

    def test_a_path_through_an_existing_list_or_scalar_is_refused_not_replaced(self, files):
        # icloud.devices.id (the index forgotten) used to replace the whole device list with
        # {id: …} and report "saved".
        files.write_text(files.read_text() + "\nicloud:\n  apple_id: me@icloud.com\n  devices:\n    - id: uuid-1\n")
        ConfigManager.get_instance().reload()
        before = files.read_text()
        with pytest.raises(ConfigStoreError, match=r"'icloud.devices' is not a section in config.yaml — address one item, e.g. icloud.devices\[0\].id"):
            ConfigStore.set_value("icloud.devices.id", "uuid-2")
        with pytest.raises(ConfigStoreError, match=r"'owner_name' is not a section in config.yaml — set it by hand"):
            ConfigStore.set_value("owner_name.first", "M")
        with pytest.raises(ConfigStoreError, match=r"'tesla' is not a list in config.yaml — set it by hand"):
            ConfigStore.set_value("tesla[0].vin", "VIN")
        assert files.read_text() == before

    def test_an_empty_section_is_filled_in(self, files):
        files.write_text(files.read_text() + "\nicloud:\n")
        ConfigStore.set_value("icloud.devices[0].id", "uuid-1")
        assert _load(files)["icloud"] == {"devices": [{"id": "uuid-1"}]}

    def test_parens_index_writes_the_same_item(self, files):
        ConfigStore.set_value("tesla.vehicles(0).vin", "VIN-PARENS")
        assert _load(files)["tesla"]["vehicles"] == [{"name": "home_car", "vin": "VIN-PARENS"}]
        assert ConfigStore.is_set("tesla.vehicles(0).vin") is True

    def test_is_set_reads_through_indexes(self, files):
        assert ConfigStore.is_set("tesla.vehicles[0].vin") is True
        assert ConfigStore.is_set("tesla.vehicles[1].vin") is False
        assert ConfigStore.is_set("icloud.devices[0].id") is False
        assert ConfigStore.is_set("tesla.vehicles[0].nope") is False


@pytest.mark.unit
class TestConcurrentWrites:
    def test_parallel_writers_all_land(self, files):
        """Eight config_set calls from one LLM reply used to race on the shared .tmp file and
        on the read-modify-write; the lock makes them take turns."""
        errors = []

        def write(i):
            try:
                ConfigStore.set_value(f"tesla.vehicles[0].field{i}", f"v{i}")
            except Exception as e:   # noqa: BLE001 — collected for the assertion
                errors.append(e)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert errors == []
        vehicle = _load(files)["tesla"]["vehicles"][0]
        assert {f"field{i}": f"v{i}" for i in range(8)}.items() <= vehicle.items()


@pytest.mark.unit
class TestSetPlain:
    @pytest.mark.parametrize("key", ["owner_name", "location.home.latitude", "icloud.apple_id", "icloud.devices[0].id",
                                     "tesla.vehicles[0].name", "chat.secret_request_timeout_seconds"])
    def test_plain_keys_are_written(self, files, key):
        assert ConfigStore.set_value(key, "42") == f"{key} saved"
        assert ConfigStore.is_set(key)

    @pytest.mark.parametrize("key", ["api_keys.openai", "pager.password", "telegram_feed.api_hash",
                                     "tesla.tessie_access_token"])
    def test_credential_looking_keys_are_written_too(self, files, key):
        """An API key the assistant read off a web page goes straight in (2026-09-16)."""
        assert ConfigStore.set_value(key, "x") == f"{key} saved"
        assert ConfigStore.is_set(key)

@pytest.mark.unit
class TestIsSetForLists:
    def test_an_empty_list_is_not_set_and_an_appended_one_is(self, files):
        files.write_text(files.read_text() + "\nicloud:\n  apple_id: me@icloud.com\n  devices: []\n")
        ConfigManager.reset_instance()
        assert ConfigStore.is_set("icloud.devices") is False
        assert ConfigStore.summary(["icloud"]) == "icloud.apple_id: set, icloud.devices: 0 devices"
        ConfigStore.set_value("icloud.devices[0].id", "uuid-1")
        assert ConfigStore.is_set("icloud.devices") is True


@pytest.mark.unit
class TestScalarLists:
    def test_index_equal_to_length_appends_a_taken_index_is_refused(self, files):
        assert ConfigStore.set_value("enabled_integrations[0]", "Tesla") == "enabled_integrations[0] saved"
        ConfigStore.set_value("enabled_integrations[1]", "Sonos & Spotify")
        assert yaml.safe_load(files.read_text())["enabled_integrations"] == ["Tesla", "Sonos & Spotify"]
        assert ConfigManager.get_instance().enabled_integrations() == ["Tesla", "Sonos & Spotify"]
        # The LLM reads one listing and enables several names from it with the same slot
        # number: the second write must not push the first name out.
        with pytest.raises(ConfigStoreError, match=r"\[1\] already holds another value — \[2\] adds the next one, or use enable_integration"):
            ConfigStore.set_value("enabled_integrations[1]", "Ring")
        assert yaml.safe_load(files.read_text())["enabled_integrations"] == ["Tesla", "Sonos & Spotify"]
        # a name already there is a no-op wherever it is addressed
        assert ConfigStore.set_value("enabled_integrations[0]", "Sonos & Spotify") == "enabled_integrations[0] saved"
        assert yaml.safe_load(files.read_text())["enabled_integrations"] == ["Tesla", "Sonos & Spotify"]
        with pytest.raises(ConfigStoreError, match=r"has 2 item\(s\) — \[2\] adds the next one"):
            ConfigStore.set_value("enabled_integrations[5]", "X")

    def test_the_list_itself_and_section_list_items_are_still_refused(self):
        with pytest.raises(ConfigStoreError, match=r"is a list of values — add one as enabled_integrations\[0\]"):
            ConfigStore.check_key("enabled_integrations")
        with pytest.raises(ConfigStoreError):
            ConfigStore.check_key("icloud.devices[0]")
        assert ConfigSchema.is_scalar_list("enabled_integrations") is True
        assert ConfigSchema.is_scalar_list("icloud.devices") is False
        assert ConfigSchema.is_scalar_list("icloud") is False

    def test_the_refusal_names_the_next_free_index(self, files):
        ConfigStore.set_value("local_outposts.exclude_mics[0]", "Acme Headset")
        with pytest.raises(ConfigStoreError, match=r"add one as local_outposts.exclude_mics\[1\]"):
            ConfigStore.check_key("local_outposts.exclude_mics")
        assert yaml.safe_load(files.read_text())["local_outposts"]["exclude_mics"] == ["Acme Headset"]
        assert ConfigManager.get_instance().get("local_outposts.exclude_mics") == ["Acme Headset"]

    def test_a_single_value_written_by_hand_is_promoted_to_a_list(self, files):
        # 2026-09-15: the LLM once stored `exclude_mics: Acme Headset` as text. The next write
        # must extend it, not refuse it, and the refusal counts it as one item.
        files.write_text(files.read_text() + "\nlocal_outposts:\n  exclude_mics: Acme Headset\n")
        ConfigManager.reset_instance()
        with pytest.raises(ConfigStoreError, match=r"add one as local_outposts.exclude_mics\[1\]"):
            ConfigStore.check_key("local_outposts.exclude_mics")
        ConfigStore.set_value("local_outposts.exclude_mics[1]", "AirPods")
        assert yaml.safe_load(files.read_text())["local_outposts"]["exclude_mics"] == ["Acme Headset", "AirPods"]
        # a list of sections is never promoted from a string
        files.write_text(files.read_text() + "\nicloud:\n  devices: junk\n")
        ConfigManager.reset_instance()
        with pytest.raises(ConfigStoreError, match="is not a list in config.yaml"):
            ConfigStore.set_value("icloud.devices[0].id", "uuid-1")


@pytest.mark.unit
class TestEnableIntegration:
    def _enabled(self, files):
        return yaml.safe_load(files.read_text()).get("enabled_integrations")

    def test_enable_appends_by_name_and_never_displaces(self, files):
        assert ConfigStore.enable_integration("iMessage") == "iMessage enabled"
        assert ConfigStore.enable_integration("Ring") == "Ring enabled"
        assert ConfigStore.enable_integration("Photos") == "Photos enabled"
        assert self._enabled(files) == ["iMessage", "Ring", "Photos"]
        assert ConfigManager.get_instance().enabled_integrations() == ["iMessage", "Ring", "Photos"]

    def test_any_spelling_already_listed_is_a_no_op(self, files):
        ConfigStore.enable_integration("Sonos & Spotify")
        assert ConfigStore.enable_integration("sonos-and-spotify") == "sonos-and-spotify was already enabled"
        assert ConfigStore.enable_integration("SonosSpotify") == "SonosSpotify was already enabled"
        assert self._enabled(files) == ["Sonos & Spotify"]

    def test_a_single_name_written_by_hand_becomes_a_list(self, files):
        files.write_text(files.read_text() + "enabled_integrations: Tesla\n")
        ConfigManager.get_instance().reload()
        ConfigStore.enable_integration("Ring")
        assert self._enabled(files) == ["Tesla", "Ring"]

    def test_disable_removes_every_spelling(self, files):
        ConfigStore.enable_integration("Ring")
        ConfigStore.enable_integration("Tesla")
        files.write_text(files.read_text().replace("- Tesla", "- Tesla\n  - tesla"))
        ConfigManager.get_instance().reload()
        assert ConfigStore.disable_integration("Tesla") == "Tesla disabled"
        assert self._enabled(files) == ["Ring"]
        assert ConfigStore.disable_integration("Tesla") == "Tesla was not enabled"

@pytest.mark.unit
class TestRenameKeys:
    OLD = ("enabled_integrations: [iMessage, telegram-monitor]\n"
           "# the bot\n"
           "telegram:\n"
           "  bot_token: '1:abc'   # secret\n"
           "feeds:\n"
           "  imessage: 60\n"
           "  telegram_monitor: 1   # short\n"
           "interfaces:\n"
           "  telegram: true\n")
    RENAMES = {"telegram": "telegram_bot", "feeds.telegram_monitor": "feeds.telegram_feed", "absent": "gone"}

    def test_renames_in_place_keeping_order_comments_and_the_rest(self, files):
        files.write_text(self.OLD)
        ConfigManager.get_instance().reload()
        done = ConfigStore.rename_keys(self.RENAMES, {"Telegram Monitor": "Telegram Feed"})
        assert done == ["telegram → telegram_bot", "feeds.telegram_monitor → feeds.telegram_feed",
                        "enabled_integrations: telegram-monitor → Telegram Feed"]
        text = files.read_text()
        assert text == (self.OLD.replace("telegram:\n  bot", "telegram_bot:\n  bot")
                        .replace("  telegram_monitor: 1   #", "  telegram_feed: 1      #")   # the comment keeps its column
                        .replace("telegram-monitor", "Telegram Feed"))
        assert ConfigManager.get_instance().get("telegram_bot.bot_token") == "1:abc"

    def test_nothing_to_rename_leaves_the_file_untouched(self, files, tmp_path):
        before = files.read_text()
        assert ConfigStore.rename_keys(self.RENAMES, {"Telegram Monitor": "Telegram Feed"}) == []
        assert files.read_text() == before
        assert not (tmp_path / "backups").exists()

    def test_old_key_stays_when_the_new_one_already_exists(self, files):
        files.write_text("telegram:\n  bot_token: old\ntelegram_bot:\n  bot_token: new\n")
        ConfigManager.get_instance().reload()
        assert ConfigStore.rename_keys(self.RENAMES) == []
        assert _load(files) == {"telegram": {"bot_token": "old"}, "telegram_bot": {"bot_token": "new"}}
