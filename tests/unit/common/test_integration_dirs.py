import contextlib

import pytest

from common import IntegrationDirs
from common.ConfigManager import config
from common.RequirementsChecker import RequirementsChecker


def _folder(root, name, *files):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "__init__.py").write_text("")
    for file in files:
        (folder / file).write_text("x")
    return folder


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """A repository integrations dir with `tesla` and `weather`, and the config listing *dirs*."""
    repo = tmp_path / "repo" / "integrations"
    _folder(repo, "tesla", "toolkit.md")
    _folder(repo, "weather")
    monkeypatch.setattr(IntegrationDirs, "INTEGRATIONS_DIR", repo)
    monkeypatch.setattr(IntegrationDirs, "AISERVER_DIR", tmp_path / "repo")
    real_get = config.get
    listed = []
    monkeypatch.setattr(config, "get", lambda key, *a, **kw: listed if key == IntegrationDirs.EXTRA_DIRS_KEY
                        else real_get(key, *a, **kw))
    IntegrationDirs.reset()
    yield tmp_path, listed
    IntegrationDirs.reset()


def test_a_listed_directory_is_a_root_and_its_folders_are_found_after_the_repositorys(layout):
    tmp_path, listed = layout
    local = tmp_path / "KeepLocal"
    _folder(local, "podcasts", "toolkit.md")
    listed.append(str(local))

    assert IntegrationDirs.roots()[1:] == (local.resolve(),)
    assert [p.parent.name for p in IntegrationDirs.files("toolkit.md")] == ["tesla", "podcasts"]
    assert [p.parent.name for p in IntegrationDirs.files("toolkit.md", local_only=True)] == ["podcasts"]


def test_a_relative_entry_is_read_from_the_repository_root(layout):
    tmp_path, listed = layout
    _folder(tmp_path / "repo" / "local", "podcasts")
    listed.append("local")

    assert IntegrationDirs.extra_roots() == ((tmp_path / "repo" / "local").resolve(),)


def test_missing_directories_and_integration_folders_listed_by_themselves_are_left_out(layout):
    tmp_path, listed = layout
    single = _folder(tmp_path / "KeepLocal", "podcasts")
    listed.extend([str(tmp_path / "nowhere"), str(single)])

    assert IntegrationDirs.extra_roots() == ()


def test_a_local_folder_named_like_a_repository_folder_is_shadowed(layout):
    tmp_path, listed = layout
    local = tmp_path / "KeepLocal"
    _folder(local, "tesla", "toolkit.md")
    _folder(local, "podcasts", "toolkit.md")
    listed.append(str(local))

    found = IntegrationDirs.files("toolkit.md")
    assert [p.parent for p in found] == [tmp_path / "repo" / "integrations" / "tesla", local.resolve() / "podcasts"]


def test_a_directory_without_an_init_is_not_an_integration_folder(layout):
    tmp_path, listed = layout
    local = tmp_path / "KeepLocal"
    (local / "notes").mkdir(parents=True)
    (local / "notes" / "toolkit.md").write_text("x")
    listed.append(str(local))

    assert IntegrationDirs.files("toolkit.md", local_only=True) == []


def test_a_local_folders_requirements_are_a_group_pip_installs(layout, monkeypatch):
    tmp_path, listed = layout
    local = tmp_path / "KeepLocal"
    requirements = _folder(local, "podcasts", "requirements.txt") / "requirements.txt"
    listed.append(str(local))
    commands = []
    monkeypatch.setattr(RequirementsChecker, "_log_tail", staticmethod(lambda lines=3: ""))
    monkeypatch.setattr("common.RequirementsChecker.PIP_LOG", tmp_path / "pip.log")
    monkeypatch.setattr("common.RequirementsChecker.subprocess.run",
                        lambda cmd, **kw: commands.append(cmd) or type("R", (), {"returncode": 0})())

    assert requirements.resolve() in [path for _, path in RequirementsChecker._groups()]
    assert RequirementsChecker.install() == ""
    assert str(requirements.resolve()) in commands[0]


def test_a_config_without_the_key_has_no_extra_roots(monkeypatch):
    """An install whose config.yaml predates `extra_integration_dirs` boots: the server read it
    before anything else and a missing key used to stop the start."""
    monkeypatch.setattr(config, "_config", {"owner_name": "Marcus"})
    IntegrationDirs.reset()
    try:
        assert IntegrationDirs.extra_roots() == ()
    finally:
        IntegrationDirs.reset()


@contextlib.contextmanager
def patch_value(value):
    """The config answering `value` itself for the key (the `layout` fixture answers a list)."""
    real_get = config.get
    config.get = lambda key, *a, **kw: value if key == IntegrationDirs.EXTRA_DIRS_KEY else real_get(key, *a, **kw)
    IntegrationDirs.reset()
    try:
        yield
    finally:
        config.get = real_get
        IntegrationDirs.reset()


def test_one_directory_given_as_a_string_is_the_one_entry(layout):
    """`extra_integration_dirs: ~/KeepLocal` reads as the one-item list it means."""
    tmp_path, listed = layout
    local = tmp_path / "KeepLocal"
    _folder(local, "podcasts")
    with patch_value(str(local)):
        assert IntegrationDirs.extra_roots() == (local.resolve(),)


def test_a_value_that_is_not_a_list_is_ignored_and_logged(layout, monkeypatch):
    tmp_path, listed = layout
    errors = []
    monkeypatch.setattr(IntegrationDirs, "error", errors.append)
    with patch_value({"dir": str(tmp_path / "KeepLocal")}):
        assert IntegrationDirs.extra_roots() == ()
    assert len(errors) == 1 and IntegrationDirs.EXTRA_DIRS_KEY in errors[0]


def test_a_local_folders_config_fragment_follows_the_repositorys_in_the_schema(layout, monkeypatch):
    """`config_help` and the key checks see a local section as they see a repository one."""
    from common import ConfigSchema
    tmp_path, listed = layout
    local = tmp_path / "KeepLocal"
    _folder(local, "podcasts")
    (local / "podcasts" / ConfigSchema.FRAGMENT_NAME).write_text("podcasts:\n  feeds: [one]\n")
    repo_fragment = tmp_path / "repo" / ConfigSchema._FRAGMENTS_FROM_ROOT / "weather" / ConfigSchema.FRAGMENT_NAME
    repo_fragment.parent.mkdir(parents=True)
    repo_fragment.write_text("weather:\n  city: Kiruna\n")
    (tmp_path / "repo" / "config.example.yaml").write_text("vault:\n  key: CHANGE_ME\n")
    monkeypatch.setattr(ConfigSchema, "EXAMPLE_PATH", tmp_path / "repo" / "config.example.yaml")
    listed.append(str(local))
    ConfigSchema.reset()
    try:
        assert ConfigSchema.sections() == ["vault", "weather", "podcasts"]
        assert ConfigSchema.is_scalar_list("podcasts.feeds")
    finally:
        ConfigSchema.reset()


def test_a_local_folders_toolkit_is_discovered_after_the_repositorys(layout):
    from assistant.commands import Toolkits
    tmp_path, listed = layout
    local = tmp_path / "KeepLocal"
    _folder(local, "podcasts")
    (local / "podcasts" / "toolkit.md").write_text(
        "---\nname: podcasts\nsummary: the podcast feeds\ncommands: [play_podcast]\n---\nText.\n")
    (tmp_path / "repo" / "integrations" / "tesla" / "toolkit.md").write_text(
        "---\nname: tesla\nsummary: the car\ncommands: [car_charge]\n---\nText.\n")
    listed.append(str(local))

    assert [t.name for t in Toolkits.discover()] == ["tesla", "podcasts"]
    assert Toolkits.discover()[1].folder == "podcasts"
